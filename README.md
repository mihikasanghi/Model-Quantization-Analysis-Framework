# Model Quantization Analysis Framework
This project implements and analyzes various quantization techniques for language models using the bitsandbytes library. It includes comprehensive testing of different quantization approaches and their impact on model performance, size, and inference speed.

## Project Structure
```
quantization-analysis/
├── results/                 # Directory for saved results
├── report.pdf         # Project report
├── q1.py                 # Main execution script
└── q2.py                 # Main execution script
```

## Dependencies
- Python 3.8+
- PyTorch 2.0+
- transformers
- bitsandbytes
- datasets
- numpy
- tabulate

## Usage

1. Run the complete analysis:
```bash
python q1.py
python q2.py
```

## Configuration

The following parameters can be modified in `config.py`:
- Model selection
- Dataset choice
- Quantization parameters
- Inference settings

## Experiments

The project includes three main quantization approaches:
1. 8-bit linear quantization
2. 4-bit linear quantization
3. 4-bit NF4 quantization (non-linear)

Each approach is evaluated on:
- Model size reduction
- Inference latency
- Perplexity impact

## Results

Results are saved in the `results/` directory with the following structure:
- JSON files containing raw metrics
- Model checkpoints (https://drive.google.com/drive/folders/1B3UYNLZNXdRGib9DEmUHd6acUf8LdAdj?usp=sharing)
- Performance summaries

Example results format:
```json
{
    "name": "quantization_type",
    "size": "model_size_mb",
    "latency": "avg_inference_ms",
    "perplexity": "perplexity_score"
}
```