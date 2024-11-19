import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import bitsandbytes as bnb
from tabulate import tabulate
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model loading and quantization"""
    name: str
    bits: Optional[int] = None
    nf4: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class InferenceResults:
    """Store inference results"""
    name: str
    sentences: List[str]
    losses: List[float]
    perplexities: List[float]
    time_taken: List[float]
    model_size: float

class DatasetLoader:
    """Handle dataset loading and preprocessing"""
    
    def __init__(self, dataset_name: str = 'ptb'):
        self.dataset_name = dataset_name
        
    def load_dataset(self) -> tuple:
        """Load and return the specified dataset"""
        if self.dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
            type_name = 'text'
        else:
            dataset = load_dataset('ptb_text_only')
            type_name = 'sentence'
            
        return dataset['test'], type_name

class ModelLoader:
    """Handle model loading and quantization"""
    
    def __init__(self, model_name: str = 'gpt2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_base_model(self) -> AutoModelForCausalLM:
        """Load the base model without quantization"""
        return AutoModelForCausalLM.from_pretrained(self.model_name)
    
    def load_quantized_model(self, config: ModelConfig) -> AutoModelForCausalLM:
        """Load a quantized version of the model"""
        if config.bits not in [4, 8]:
            raise ValueError("Quantization bits must be either 4 or 8")
            
        quant_config = self._get_quantization_config(config)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config
        )
        
        self._verify_quantization(model, config)
        return model
    
    def _get_quantization_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Get the appropriate quantization configuration"""
        if config.bits == 8:
            return {
                "load_in_8bit": True,
                "bnb_8bit_compute_dtype": torch.float16
            }
        else:
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4" if config.nf4 else "fp4",
                "bnb_4bit_use_double_quant": config.nf4,
                "bnb_4bit_compute_dtype": torch.float16
            }
    
    def _verify_quantization(self, model: AutoModelForCausalLM, config: ModelConfig) -> None:
        """Verify the quantization status of the model"""
        modules_quantized = sum(
            1 for module in model.modules() 
            if isinstance(module, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit))
        )
        total_modules = sum(
            1 for module in model.modules() 
            if isinstance(module, torch.nn.Linear)
        )
        
        logger.info(f"\nQuantization Summary: {config.bits}-bit")
        if config.bits == 4:
            logger.info(f"Quantization type: {'NF4' if config.nf4 else 'FP4'}")
        logger.info(f"Modules quantized: {modules_quantized}/{total_modules}")
        logger.info(f"Approximate memory savings: {(1 - config.bits/32) * 100:.1f}%")

class InferenceEngine:
    """Handle model inference and results collection"""
    
    def __init__(self, save_dir: str = './results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def run_inference(self, 
                     config: ModelConfig, 
                     model: AutoModelForCausalLM, 
                     tokenized_data: List[Dict[str, torch.Tensor]]) -> InferenceResults:
        """Run inference and collect results"""
        model.eval()
        if config.name == 'No Quantization':
            model.to(config.device)
            
        results = []
        total_samples = len(tokenized_data)
        
        for idx, batch in enumerate(tokenized_data, 1):
            logger.info(f'Processing sample {idx}/{total_samples}')
            result = self._process_single_batch(batch, model, config.device)
            results.append(result)
            
        return self._compile_results(config.name, model, results)
    
    def _process_single_batch(self, 
                            batch: Dict[str, torch.Tensor], 
                            model: AutoModelForCausalLM, 
                            device: str) -> Dict[str, Any]:
        """Process a single batch of data"""
        input_ids = batch['input_ids'].to(device).unsqueeze(0)
        attention_mask = batch['attention_mask'].to(device).unsqueeze(0)
        labels = batch['input_ids'].to(device).unsqueeze(0)
        
        with torch.no_grad():
            start_time = time.time()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            inference_time = time.time() - start_time
            
        return {
            'sentence': batch['original_sentence'],
            'loss': output.loss.item(),
            'perplexity': np.exp(output.loss.item()),
            'time_taken': inference_time
        }
    
    def _compile_results(self, 
                        name: str, 
                        model: AutoModelForCausalLM, 
                        results: List[Dict[str, Any]]) -> InferenceResults:
        """Compile all results into a single object"""
        model_size = sum(
            param.numel() * param.element_size() 
            for param in model.state_dict().values()
        ) / (1024 ** 2)
        
        inference_results = InferenceResults(
            name=name,
            sentences=[r['sentence'] for r in results],
            losses=[r['loss'] for r in results],
            perplexities=[r['perplexity'] for r in results],
            time_taken=[r['time_taken'] for r in results],
            model_size=model_size
        )
        
        self._save_results(inference_results)
        return inference_results
    
    def _save_results(self, results: InferenceResults) -> None:
        """Save results to disk"""
        results_dict = {
            "name": results.name,
            "sentences": results.sentences,
            "losses": results.losses,
            "perplexities": results.perplexities,
            "time_taken": results.time_taken,
            "model_size": results.model_size
        }
        
        results_path = self.save_dir / f"{results.name.replace(' ', '_')}_results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"Results saved to {results_path}")

class ResultsAnalyzer:
    """Analyze and display results"""
    
    @staticmethod
    def display_results(results: List[InferenceResults]) -> None:
        """Display results in a formatted table"""
        headers = ["Type", "Size (MB)", "Average Inference Latency (ms)", "Average Perplexity"]
        table = [
            [
                res.name,
                round(res.model_size, 2),
                round(np.nanmean(res.time_taken) * 1000, 2),
                round(np.exp(np.nanmean(res.losses)), 2)
            ]
            for res in results
        ]
        
        print("\nResults Summary:")
        print(tabulate(table, headers=headers, tablefmt="grid"))

def main():
    # Initialize components
    dataset_loader = DatasetLoader()
    model_loader = ModelLoader()
    inference_engine = InferenceEngine()
    
    # Load dataset
    test_data, type_name = dataset_loader.load_dataset()
    
    # Tokenize data
    tokenized_data = [{
        **model_loader.tokenizer(
            x[type_name], 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ),
        'original_sentence': x[type_name]
    } for x in test_data]
    
    # Define model configurations
    configs = [
        ModelConfig(name="No Quantization"),
        ModelConfig(name="8-bit", bits=8),
        ModelConfig(name="4-bit", bits=4),
        ModelConfig(name="NF4", bits=4, nf4=True)
    ]
    
    # Run experiments
    results = []
    for config in configs:
        logger.info(f"\nRunning inference for {config.name}")
        model = (model_loader.load_base_model() if config.bits is None 
                else model_loader.load_quantized_model(config))
        result = inference_engine.run_inference(config, model, tokenized_data)
        results.append(result)
    
    # Display results
    ResultsAnalyzer.display_results(results)

if __name__ == "__main__":
    main()