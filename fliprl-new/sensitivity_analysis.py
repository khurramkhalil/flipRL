import torch
import numpy as np
import csv
import time
from transformers import AutoModelForCausalLM, AutoTokenizer # Use AutoModel for flexibility
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt # For plotting
import struct # For accurate float bit manipulation

def float_to_bits(f):
    """ Convert float to its 32-bit integer representation """
    s = struct.pack('>f', f)
    return struct.unpack('>I', s)[0]

def bits_to_float(b):
    """ Convert 32-bit integer representation back to float """
    s = struct.pack('>I', b)
    return struct.unpack('>f', s)[0]

def flip_lsb_float(f):
    """ Flips the Least Significant Bit of a 3.4e38 float """
    bits = float_to_bits(f)
    flipped_bits = bits ^ 1 # XOR with 1 flips the LSB
    return bits_to_float(flipped_bits)

def flip_lsb_int(i, num_bits=8):
    """ Flips the Least Significant Bit of an integer """
    mask = 1
    return i ^ mask

# Helper function to get nested layers (handles models like LLaMA)
def get_transformer_layers(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h # GPT-2 style
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers # LLaMA style
    # Add more checks for other model architectures if needed
    else:
        raise ValueError("Could not automatically detect transformer layers in the model structure.")

def analyze_layer_sensitivity(model_name="gpt2-large", output_file="layer_sensitivity.csv",
                              selection_rate=0.001, num_examples=10, device=None,
                              alpha=0.5, batch_size=4, plot_results=True, is_quantized=False):
    """
    Analyze layer sensitivity using a hybrid metric and LSB flips.

    Args:
        model_name: The pretrained model name or path.
        output_file: Path to output CSV file.
        selection_rate: Percentage (0 to 1) of top sensitive parameters to flip.
        num_examples: Number of test examples for gradient/loss calculation.
        device: Device ('cuda' or 'cpu').
        alpha: Mixing coefficient for hybrid sensitivity (0=magnitude, 1=gradient).
        batch_size: Batch size for evaluation.
        plot_results: Whether to generate a plot of layer sensitivity.
        is_quantized: Set True if the model weights are quantized integers (e.g., INT8).
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running layer sensitivity analysis on {model_name} using {device}")
    print(f"Config: alpha={alpha}, selection_rate={selection_rate*100:.3f}%, num_examples={num_examples}, batch_size={batch_size}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if missing
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print("Added [PAD] token.")

    model.eval()

    # Get transformer layers
    try:
        layers = get_transformer_layers(model)
        num_layers = len(layers)
        print(f"Model has {num_layers} transformer layers.")
    except ValueError as e:
        print(f"Error: {e}")
        return None

    # Create simple test examples
    # Using slightly longer/varied examples for better gradient signal
    test_inputs_raw = [
        "The quick brown fox jumps over the lazy",
        "Artificial intelligence is rapidly evolving field with applications in",
        "Large language models demonstrate impressive capabilities in text generation and",
        "Machine learning algorithms learn patterns from",
        "Binary code consists of zeros and",
        "To learn programming effectively, one should practice consistently and build",
        "Neural networks are composed of interconnected nodes or neurons organized in",
        "Data science involves extracting knowledge and insights from structured and unstructured",
        "The internet originated from research funded by the US government in the",
        "Quantum computing leverages principles of quantum mechanics like superposition and"
    ] * ( (num_examples // 10) + 1) # Repeat to get enough examples

    test_inputs_raw = test_inputs_raw[:num_examples]

    # Tokenize inputs for batch processing
    print("Tokenizing and preparing evaluation batch...")
    inputs = tokenizer(test_inputs_raw, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
    # Use input_ids as labels for language modeling loss
    inputs['labels'] = inputs.input_ids.clone()

    # Function to evaluate loss on the prepared batch
    def calculate_batch_loss(model, batch_inputs):
        model.eval() # Ensure evaluation mode
        with torch.no_grad():
            outputs = model(**batch_inputs)
            loss = outputs.loss.item() # Get scalar loss value
        return loss

    # Calculate baseline loss
    print("Calculating baseline loss...")
    baseline_loss = calculate_batch_loss(model, inputs)
    print(f"Baseline average loss: {baseline_loss:.4f}")

    # Results container
    layer_results = []

    # --- Analyze each layer ---
    print("Analyzing layer sensitivity...")
    # Wrap layers access with tqdm for progress bar
    for layer_idx, layer in enumerate(tqdm(layers, desc="Analyzing Layers")):

        # --- 1. Get Layer Parameters and Calculate Gradients ---
        model.zero_grad() # Clear previous gradients
        layer_params = {name: param for name, param in layer.named_parameters() if param.requires_grad}
        original_state_dict = {name: param.data.clone() for name, param in layer_params.items()}

        # Forward pass to get outputs (needed for backward pass)
        # We need gradients w.r.t. the batch loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass to calculate gradients for parameters in this layer
        loss.backward()

        # --- 2. Calculate Sensitivity Score ---
        param_sensitivities = []
        param_names = []
        param_shapes = {}
        flat_indices_map = {} # Map flat index back to (name, tensor_index)
        current_offset = 0

        with torch.no_grad():
            for name, param in layer_params.items():
                if param.grad is None:
                    print(f"Warning: No gradient for param {name} in layer {layer_idx}. Skipping.")
                    continue

                # Clone to avoid modifying original grads/params
                weights = param.data.float().clone() # Ensure float for calculations
                grads = param.grad.float().clone()

                # L2 Normalization (handle potential zero norms)
                norm_weights = torch.linalg.norm(weights.flatten().float())
                norm_grads = torch.linalg.norm(grads.flatten().float())

                normalized_weights = weights / (norm_weights + 1e-12) # Add epsilon for stability
                normalized_grads = grads / (norm_grads + 1e-12)

                # Calculate hybrid sensitivity score (element-wise)
                sensitivity = alpha * normalized_grads.abs() + (1 - alpha) * normalized_weights.abs()

                # Store flattened sensitivities and meta-data
                flat_sensitivity = sensitivity.flatten()
                param_sensitivities.append(flat_sensitivity)
                param_names.extend([name] * flat_sensitivity.numel())
                param_shapes[name] = param.shape

                # Build reverse map
                for i in range(flat_sensitivity.numel()):
                    flat_indices_map[current_offset + i] = (name, np.unravel_index(i, param.shape))
                current_offset += flat_sensitivity.numel()

        if not param_sensitivities:
             print(f"Layer {layer_idx} has no parameters with gradients. Skipping.")
             model.zero_grad() # Clean up grads
             continue

        all_sensitivities = torch.cat(param_sensitivities)
        total_params_in_layer = all_sensitivities.numel()

        # --- 3. Select Top-k Parameters ---
        k = int(selection_rate * total_params_in_layer)
        if k == 0 and total_params_in_layer > 0:
            k = 1 # Ensure at least one flip if possible

        if k > 0:
            _, top_k_flat_indices = torch.topk(all_sensitivities, k)
        else:
            top_k_flat_indices = torch.tensor([], dtype=torch.long, device=device) # Handle case with 0 params

        # --- 4. Flip LSB of Selected Parameters ---
        perturbed_state_dict = {name: tensor.clone() for name, tensor in original_state_dict.items()} # Start with original values
        num_actually_flipped = 0

        with torch.no_grad():
            for flat_idx_tensor in top_k_flat_indices:
                flat_idx = flat_idx_tensor.item()
                if flat_idx in flat_indices_map:
                    name, tensor_indices = flat_indices_map[flat_idx]
                    original_value = perturbed_state_dict[name][tensor_indices]

                    # Apply appropriate flip based on type
                    if is_quantized or 'int' in str(original_value.dtype):
                         # Assuming integer type if is_quantized or dtype hints at it
                         num_bits = original_value.element_size() * 8 # Get num bits from dtype
                         flipped_value = flip_lsb_int(original_value.item(), num_bits)
                         perturbed_state_dict[name][tensor_indices] = torch.tensor(flipped_value, dtype=original_value.dtype)
                    elif 'float' in str(original_value.dtype) or 'bfloat' in str(original_value.dtype):
                         # Use float LSB flip (handle potential errors if not standard float32)
                         try:
                            flipped_value = flip_lsb_float(original_value.item())
                            perturbed_state_dict[name][tensor_indices] = torch.tensor(flipped_value, dtype=original_value.dtype)
                         except struct.error:
                             # Fallback for non-32bit floats (e.g., bfloat16): small perturbation
                             print(f"Warning: Cannot perform precise LSB flip on {original_value.dtype}. Using perturbation.")
                             eps = torch.finfo(original_value.dtype).eps * 10 # Small perturbation
                             perturbed_state_dict[name][tensor_indices] += eps
                    else:
                         print(f"Warning: Unsupported dtype {original_value.dtype} for LSB flip. Skipping param {name}.")
                         continue # Skip if type is unknown

                    num_actually_flipped += 1
                else:
                    print(f"Warning: Flat index {flat_idx} not found in map.")


        # --- 5. Evaluate Performance with Flipped Bits ---
        # Temporarily load the perturbed weights into the model layer
        current_layer_state = {name: param.data.clone() for name, param in layer_params.items()} # Save current state before loading perturbed
        try:
            with torch.no_grad():
                for name, perturbed_tensor in perturbed_state_dict.items():
                     if name in layer_params:
                         layer_params[name].data.copy_(perturbed_tensor)

            perturbed_loss = calculate_batch_loss(model, inputs)
            loss_increase = perturbed_loss - baseline_loss

        except Exception as e:
            print(f"Error during evaluation of layer {layer_idx} with perturbed weights: {e}")
            perturbed_loss = baseline_loss # Assume no change if error
            loss_increase = 0

        finally:
            # --- 6. Restore Original Weights ---
            # IMPORTANT: Always restore original weights
             with torch.no_grad():
                for name, original_tensor in original_state_dict.items():
                    if name in layer_params:
                         layer_params[name].data.copy_(original_tensor)
        # Clear gradients for the next iteration
        model.zero_grad()


        # Store results
        layer_results.append({
            'layer_idx': layer_idx,
            'baseline_loss': baseline_loss,
            'perturbed_loss': perturbed_loss,
            'loss_increase': loss_increase,
            'num_flipped_bits': num_actually_flipped
        })

        # Optional: print progress
        # print(f"Layer {layer_idx}: Loss increase: {loss_increase:.4f}, Flipped: {num_actually_flipped}")


    # --- Post-Analysis ---
    # Find most sensitive layer based on loss increase
    if layer_results:
        # Sort layers by loss increase (higher is more sensitive)
        layer_results_sorted = sorted(layer_results, key=lambda x: x['loss_increase'], reverse=True)
        most_sensitive = layer_results_sorted[0]
        print(f"\nMost sensitive layer: {most_sensitive['layer_idx']} "
              f"with loss increase of {most_sensitive['loss_increase']:.4f} "
              f"(flipped {most_sensitive['num_flipped_bits']} bits)")

        # Write results to CSV
        print(f"Writing results to {output_file}")
        try:
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['layer_idx', 'baseline_loss', 'perturbed_loss', 'loss_increase', 'num_flipped_bits']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # Write sorted results (most sensitive first)
                for result in layer_results_sorted:
                    writer.writerow(result)
        except IOError as e:
            print(f"Error writing CSV file: {e}")

        # Plot results if requested
        if plot_results:
            try:
                layer_indices = [r['layer_idx'] for r in layer_results]
                loss_increases = [r['loss_increase'] for r in layer_results]

                plt.figure(figsize=(12, 6))
                plt.bar(layer_indices, loss_increases, color='skyblue')
                plt.xlabel("Layer Index")
                plt.ylabel("Loss Increase")
                plt.title(f"Layer Sensitivity Analysis for {model_name}\n(Impact of {selection_rate*100:.3f}% LSB Flips based on Hybrid Sensitivity α={alpha})")
                plt.xticks(np.arange(min(layer_indices), max(layer_indices)+1, step=max(1, num_layers // 10))) # Adjust x-ticks
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plot_filename = output_file.replace(".csv", ".png")
                plt.savefig(plot_filename)
                print(f"Sensitivity plot saved to {plot_filename}")
                # plt.show() # Optionally display the plot
            except Exception as e:
                 print(f"Error generating plot: {e}")

    else:
        print("No valid layer results obtained.")

    return layer_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Layer Sensitivity of LLMs to Bit Flips")
    parser.add_argument("--model", type=str, default="gpt2-large", help="Hugging Face model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--output", type=str, default="layer_sensitivity_results.csv", help="Output CSV file path")
    parser.add_argument("--selection-rate", type=float, default=0.001, help="Fraction (0 to 1) of top sensitive parameters to flip per layer (e.g., 0.001 for 0.1%)")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples for calculating loss/gradients")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation runs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid sensitivity mixing coefficient (0=magnitude only, 1=gradient only)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu', defaults to cuda if available)")
    parser.add_argument("--quantized", action='store_true', help="Set if the model uses quantized integer weights")
    parser.add_argument("--no-plot", action='store_true', help="Disable automatic plotting of results")

    args = parser.parse_args()

    start_time = time.time()
    analyze_layer_sensitivity(
        model_name=args.model,
        output_file=args.output,
        selection_rate=args.selection_rate,
        num_examples=args.num_examples,
        device=args.device,
        alpha=args.alpha,
        batch_size=args.batch_size, # Pass batch_size
        plot_results=not args.no_plot,
        is_quantized=args.quantized
    )
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")