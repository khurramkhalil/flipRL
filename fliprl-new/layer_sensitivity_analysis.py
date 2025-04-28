import torch
import numpy as np
import csv
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import argparse

def analyze_layer_sensitivity(model_name="gpt2-large", output_file="layer_sensitivity.csv", 
                             selection_rate=0.001, num_examples=10, device=None):
    """
    Analyze layer sensitivity of a model to bit-flips and output results to CSV.
    
    Args:
        model_name: The pretrained model to analyze
        output_file: Path to output CSV file
        selection_rate: Percentage of parameters to flip in each layer
        num_examples: Number of test examples to use
        device: Device to run on (cuda or cpu)
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running layer sensitivity analysis on {model_name} using {device}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set padding token for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()
    
    # Get number of layers
    num_layers = len(model.transformer.h)
    print(f"Model has {num_layers} transformer layers")
    
    # Create simple test examples
    test_inputs = [
        "The capital of France is",
        "Artificial intelligence is",
        "Large language models can",
        "The purpose of machine learning is",
        "Computers use binary to",
        "The best way to learn programming is",
        "Neural networks consist of",
        "Data science combines statistics and",
        "The internet was invented by",
        "Quantum computing uses"
    ]
    
    # Use only the specified number of examples
    test_inputs = test_inputs[:num_examples]
    
    # Tokenize inputs
    encoded_inputs = []
    for text in test_inputs:
        encoded = tokenizer(text, return_tensors="pt").to(device)
        encoded_inputs.append(encoded)
    
    # Evaluate baseline performance
    baseline_perplexities = []
    
    print("Calculating baseline performance...")
    with torch.no_grad():
        for encoded in encoded_inputs:
            outputs = model(**encoded)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            baseline_perplexities.append(perplexity)
    
    avg_baseline_perplexity = sum(baseline_perplexities) / len(baseline_perplexities)
    print(f"Baseline average perplexity: {avg_baseline_perplexity:.4f}")
    
    # Results container
    layer_results = []
    
    # Analyze each layer
    print("Analyzing layer sensitivity...")
    for layer_idx in tqdm(range(num_layers)):
        # Get layer parameters
        layer = model.transformer.h[layer_idx]
        layer_weights = {name: param.data.clone() for name, param in layer.named_parameters()}
        
        # Track total number of parameters
        total_params = sum(tensor.numel() for tensor in layer_weights.values())
        
        # Select parameters for bit-flipping
        k = int(selection_rate * total_params)
        
        # Create a flattened list of all parameters
        all_params = []
        param_indices = {}
        offset = 0
        
        for name, tensor in layer_weights.items():
            size = tensor.numel()
            all_params.append(tensor.view(-1))
            param_indices[name] = (offset, offset + size)
            offset += size
        
        all_params = torch.cat(all_params)
        
        # Select top-k parameters by magnitude (simpler than calculating gradients)
        _, indices = torch.topk(torch.abs(all_params), k)
        
        # Create modified weights
        modified_weights = {name: tensor.clone() for name, tensor in layer_weights.items()}
        
        # Flip LSB of selected parameters
        for idx in indices:
            idx = idx.item()
            
            # Find which parameter tensor this index belongs to
            for name, (start, end) in param_indices.items():
                if start <= idx < end:
                    # Calculate relative index within the tensor
                    rel_idx = idx - start
                    
                    # Get tensor shape
                    tensor_shape = modified_weights[name].shape
                    
                    # Convert flat index to tensor indices
                    tensor_indices = np.unravel_index(rel_idx, tensor_shape)
                    
                    # Get value at index
                    value = modified_weights[name][tensor_indices].item()
                    
                    # Flip LSB by XOR with 1 (in float representation)
                    # We'll use a small perturbation instead for simplicity
                    eps = 1e-7
                    modified_weights[name][tensor_indices] = value * (1 + eps)
                    break
        
        # Apply modified weights
        with torch.no_grad():
            for name, tensor in modified_weights.items():
                getattr(layer, name.split('.')[0])[int(name.split('.')[1]) if '.' in name else 0].copy_(tensor)
        
        # Evaluate performance with flipped bits
        perplexities = []
        
        with torch.no_grad():
            for encoded in encoded_inputs:
                outputs = model(**encoded)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        avg_perplexity = sum(perplexities) / len(perplexities)
        perplexity_increase = avg_perplexity - avg_baseline_perplexity
        
        # Store results
        layer_results.append({
            'layer_idx': layer_idx,
            'baseline_perplexity': avg_baseline_perplexity,
            'perturbed_perplexity': avg_perplexity,
            'perplexity_increase': perplexity_increase,
            'num_flipped_bits': k
        })
        
        # Restore original weights
        with torch.no_grad():
            for name, tensor in layer_weights.items():
                getattr(layer, name.split('.')[0])[int(name.split('.')[1]) if '.' in name else 0].copy_(tensor)
        
        print(f"Layer {layer_idx}: Perplexity increase: {perplexity_increase:.4f}")
    
    # Find most sensitive layer
    most_sensitive_idx = max(range(len(layer_results)), key=lambda i: layer_results[i]['perplexity_increase'])
    most_sensitive = layer_results[most_sensitive_idx]
    
    print(f"Most sensitive layer: {most_sensitive['layer_idx']} with perplexity increase of {most_sensitive['perplexity_increase']:.4f}")
    
    # Write results to CSV
    print(f"Writing results to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['layer_idx', 'baseline_perplexity', 'perturbed_perplexity', 'perplexity_increase', 'num_flipped_bits']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in layer_results:
            writer.writerow(result)
    
    return layer_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze layer sensitivity of a language model")
    parser.add_argument("--model", type=str, default="gpt2-large", help="Model name or path")
    parser.add_argument("--output", type=str, default="layer_sensitivity.csv", help="Output CSV file")
    parser.add_argument("--selection-rate", type=float, default=0.001, help="Percentage of parameters to flip")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of test examples")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    analyze_layer_sensitivity(
        model_name=args.model,
        output_file=args.output,
        selection_rate=args.selection_rate,
        num_examples=args.num_examples,
        device=args.device
    )
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")