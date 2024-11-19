import numpy as np 
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters"""
    num_bits: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class InferenceResults:
    """Container for inference results"""
    name: str
    sentences: List[str]
    losses: List[float]
    perplexities: List[float]
    time_taken: List[float]
    model_size: float

    def get_average_metrics(self) -> Dict[str, float]:
        return {
            "avg_loss": np.nanmean(self.losses),
            "avg_perplexity": np.exp(np.nanmean(self.losses)),
            "avg_inference_time_ms": np.nanmean(self.time_taken) * 1000
        }

    def save_results(self) -> None:
        results_path = f"{self.name.replace(' ', '_')}_results.json"
        with open(results_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

class Quantizer:
    """Base quantization class"""
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._setup_quantization_params()

    def _setup_quantization_params(self) -> None:
        n = self.config.num_bits
        n_ = n - 1
        self.qmax = 2**n_ - 1
        self.qmin = -1 * (2**n_)

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        data_min = tensor.min().item()
        data_max = tensor.max().item()
        
        scale = (data_max - data_min) / (self.qmax - self.qmin)
        scale = max(scale, 1e-8)  # Prevent divide by zero
        
        zero_point = round(self.qmin - data_min / scale)
        zero_point = int(zero_point)
        
        quantized_tensor = torch.round((tensor / scale) + zero_point).to(torch.int8)
        return quantized_tensor, scale, zero_point

    @staticmethod
    def dequantize_tensor(quantized_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        return (quantized_tensor.to(torch.float32) - zero_point) * scale

class BaseQuantizedModel(nn.Module):
    """Base class for quantized models"""
    def __init__(self, original_model: nn.Module, quantizer: Quantizer):
        super().__init__()
        self.original_model = deepcopy(original_model)
        self.quantizer = quantizer
        self.quantization_params = {}
        self.model_size = 0

    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        return self.model_size / (1024 ** 2)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        input_ids = input_ids.to(torch.long)
        return self.original_model(input_ids, **kwargs)

class FullQuantizedModel(BaseQuantizedModel):
    """Model with all parameters quantized"""
    def __init__(self, original_model: nn.Module, quantizer: Quantizer):
        super().__init__(original_model, quantizer)
        self._quantize_all_parameters()

    def _quantize_all_parameters(self) -> None:
        for name, param in self.original_model.named_parameters():
            quantized_tensor, scale, zero_point = self.quantizer.quantize_tensor(param.data)
            self.model_size += quantized_tensor.numel() * quantized_tensor.element_size()
            self.quantization_params[name] = {"scale": scale, "zero_point": zero_point}
            param.data = self.quantizer.dequantize_tensor(quantized_tensor, scale, zero_point)
        
        self.model_size = self._calculate_model_size()

class DecoderQuantizedModel(BaseQuantizedModel):
    """Model with only decoder blocks quantized"""
    def __init__(self, original_model: nn.Module, quantizer: Quantizer, quantized_blocks: List[int]):
        super().__init__(original_model, quantizer)
        self.quantized_blocks = quantized_blocks
        self._quantize_decoder_blocks()

    def _quantize_decoder_blocks(self) -> None:
        for name, param in self.original_model.named_parameters():
            is_decoder_block = any(f"transformer.h.{idx}." in name for idx in self.quantized_blocks)
            
            if is_decoder_block:
                quantized_tensor, scale, zero_point = self.quantizer.quantize_tensor(param.data)
                self.model_size += quantized_tensor.numel() * quantized_tensor.element_size()
                self.quantization_params[name] = {"scale": scale, "zero_point": zero_point}
                param.data = self.quantizer.dequantize_tensor(quantized_tensor, scale, zero_point)
            else:
                self.model_size += param.numel() * param.element_size()
        
        self.model_size = self._calculate_model_size()

class DataProcessor:
    """Handle dataset loading and tokenization"""
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def load_dataset(dataset_name: str) -> tuple:
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
            type_name = 'text'
        else:
            dataset = load_dataset('ptb_text_only')
            type_name = 'sentence'
        return dataset, type_name

    def tokenize(self, data: List[dict], type_name: str) -> List[dict]:
        return [{
            **self.tokenizer(
                x[type_name],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ),
            'original_sentence': x[type_name]
        } for x in data]

class ModelEvaluator:
    """Handle model evaluation and inference"""
    def __init__(self, config: QuantizationConfig):
        self.config = config

    def run_inference(self, model: nn.Module, data: List[dict], name: str) -> InferenceResults:
        model.eval()
        model.to(self.config.device)

        sentences, losses, perplexities, times = [], [], [], []

        print(f"Running inference for {name}...")
        for idx, sample in enumerate(data, 1):
            print(f'Sample {idx}/{len(data)}', end='\r')
            
            metrics = self._process_sample(model, sample)
            
            sentences.append(sample['original_sentence'])
            losses.append(metrics['loss'])
            perplexities.append(metrics['perplexity'])
            times.append(metrics['time'])

        results = InferenceResults(
            name=name,
            sentences=sentences,
            losses=losses,
            perplexities=perplexities,
            time_taken=times,
            model_size=self._get_model_size(model)
        )
        
        results.save_results()
        self._save_model(model, name)
        
        return results

    def _process_sample(self, model: nn.Module, sample: dict) -> Dict[str, float]:
        input_id = sample['input_ids'].to(self.config.device).unsqueeze(0)
        attention_mask = sample['attention_mask'].to(self.config.device).unsqueeze(0)
        label = sample['input_ids'].to(self.config.device).unsqueeze(0)

        with torch.no_grad():
            start_time = time.time()
            output = model(input_id, attention_mask=attention_mask, labels=label)
            inference_time = time.time() - start_time
            
            loss = output.loss.item()
            perplexity = np.exp(loss)

        return {
            'loss': loss,
            'perplexity': perplexity,
            'time': inference_time
        }

    def _get_model_size(self, model: nn.Module) -> float:
        if hasattr(model, 'model_size'):
            return model.model_size
        return sum(p.numel() * p.element_size() for p in model.state_dict().values()) / (1024 ** 2)

    def _save_model(self, model: nn.Module, name: str) -> None:
        model_path = "{name.replace(' ', '_')}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

def display_results(results: List[InferenceResults]) -> None:
    """Display evaluation results in a formatted table"""
    from tabulate import tabulate
    
    headers = ["Type", "Size (MB)", "Average Inference Latency (ms)", "Average Perplexity"]
    table_data = []
    
    for result in results:
        metrics = result.get_average_metrics()
        table_data.append([
            result.name,
            result.model_size,
            metrics["avg_inference_time_ms"],
            metrics["avg_perplexity"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    # Configuration
    config = QuantizationConfig(num_bits=8)
    
    # Initialize models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Initialize quantizer and data processor
    quantizer = Quantizer(config)
    data_processor = DataProcessor(tokenizer)
    
    # Load and process dataset
    dataset, type_name = data_processor.load_dataset('ptb_text_only')
    test_data = dataset['test']
    tokenized_test_data = data_processor.tokenize(test_data, type_name)
    
    # Initialize models
    full_quantized_model = FullQuantizedModel(base_model, quantizer)
    decoder_quantized_model = DecoderQuantizedModel(base_model, quantizer, quantized_blocks=[2,4,6])
    
    # Evaluate models
    evaluator = ModelEvaluator(config)
    results = []
    
    for model_info in [
        (base_model, "No Quantization"),
        (full_quantized_model, "Full Quantization"),
        (decoder_quantized_model, "Decoder only Quantization")
    ]:
        results.append(evaluator.run_inference(model_info[0], tokenized_test_data, model_info[1]))
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main()