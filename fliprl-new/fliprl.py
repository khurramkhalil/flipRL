import torch
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import copy
import warnings
warnings.filterwarnings("ignore")

class FlipRL:
    # Update to the __init__ function in FlipRL class to set the pad token
    def __init__(self, model_name="gpt2-large", alpha=0.5, selection_rate=0.001, 
                device="cuda" if torch.cuda.is_available() else "cpu",
                rl_learning_rate=0.1, rl_discount_factor=0.9, 
                rl_exploration_rate=0.1, rl_generations=10):
        """
        Initialize the FlipRL framework.
        
        Args:
            model_name: The name or path of the pre-trained model
            alpha: The sensitivity mixing coefficient (between gradient and magnitude)
            selection_rate: The percentage of parameters to consider in the initial subset
            device: The computation device (CPU or CUDA)
            rl_learning_rate: Learning rate for Q-learning
            rl_discount_factor: Discount factor for future rewards
            rl_exploration_rate: Exploration rate for epsilon-greedy strategy
            rl_generations: Number of RL training generations
        """
        self.model_name = model_name
        self.alpha = alpha
        self.selection_rate = selection_rate
        self.device = device
        
        # RL hyperparameters
        self.rl_alpha = rl_learning_rate
        self.rl_gamma = rl_discount_factor
        self.rl_epsilon = rl_exploration_rate
        self.rl_generations = rl_generations
        
        print(f"Initializing FlipRL with {model_name} on {device}")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set padding token for the tokenizer (necessary for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize layer information
        self.num_layers = len(self.model.transformer.h)
        print(f"Model has {self.num_layers} transformer layers")
        
        # Additional properties to be set during execution
        self.layer_sensitivity_results = []
        self.most_sensitive_layer_idx = None
        self.critical_indices = None
        self.final_accuracy = None

    def get_layer_weights(self, layer_idx):
        """Extract weights for a specific layer."""
        return self.model.transformer.h[layer_idx].state_dict()
    
    def set_layer_weights(self, layer_idx, weights):
        """Set weights for a specific layer."""
        layer = self.model.transformer.h[layer_idx]
        for name, param in weights.items():
            if name in layer.state_dict():
                layer.state_dict()[name].copy_(param)
        return self.model
    
    def calculate_gradients(self, layer_idx, eval_data):
        """Calculate gradients for a specific layer using evaluation data."""
        # Ensure model is in training mode for gradient calculation
        self.model.train()
        
        # Create a small batch from eval data
        batch_size = min(4, len(eval_data))
        batch_indices = np.random.choice(len(eval_data), batch_size, replace=False)
        batch = [eval_data[i] for i in batch_indices]
        
        # Clear existing gradients
        self.model.zero_grad()
        
        # Process each input separately to avoid batch size mismatch
        layer_gradients = {}
        accumulated_gradients = {}
        
        for item in batch:
            # Tokenize single input
            inputs = self.tokenizer(item["input"], return_tensors="pt").to(self.device)
            
            # For GPT-2, we use the same input as the target (shifted by 1)
            labels = inputs.input_ids.clone()
            
            # Forward pass
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss / batch_size  # Scale the loss by batch size for proper accumulation
            
            # Backward pass to compute gradients
            loss.backward()
        
        # Extract gradients for the specific layer
        for name, param in self.model.transformer.h[layer_idx].named_parameters():
            if param.grad is not None:
                layer_gradients[name] = param.grad.clone().detach()
        
        # Return model to evaluation mode
        self.model.eval()
        
        return layer_gradients
    
    def normalize_tensors(self, weights, gradients):
        """L2 normalize weight and gradient tensors."""
        normalized_weights = {}
        normalized_gradients = {}
        
        for name in weights:
            if name in gradients:
                # Flatten tensors for normalization
                w_flat = weights[name].view(-1)
                g_flat = gradients[name].view(-1)
                
                # L2 normalization
                w_norm = torch.norm(w_flat, p=2)
                g_norm = torch.norm(g_flat, p=2)
                
                # Avoid division by zero
                w_norm = w_norm if w_norm > 0 else 1.0
                g_norm = g_norm if g_norm > 0 else 1.0
                
                normalized_weights[name] = weights[name] / w_norm
                normalized_gradients[name] = gradients[name] / g_norm
        
        return normalized_weights, normalized_gradients
    
    def calculate_sensitivity_scores(self, normalized_weights, normalized_gradients):
        """Calculate sensitivity scores using the hybrid metric."""
        sensitivity_scores = {}
        
        for name in normalized_weights:
            if name in normalized_gradients:
                sensitivity_scores[name] = (
                    self.alpha * torch.abs(normalized_gradients[name]) + 
                    (1 - self.alpha) * torch.abs(normalized_weights[name])
                )
        
        return sensitivity_scores
    
    def get_top_k_indices(self, sensitivity_scores, k):
        """Get indices of top-k sensitive parameters."""
        top_indices = {}
        
        for name, scores in sensitivity_scores.items():
            # Flatten scores for ranking
            flat_scores = scores.view(-1)
            
            # Get number of parameters to select
            num_params = flat_scores.numel()
            k_for_tensor = min(k, num_params)  # Ensure k doesn't exceed tensor size
            
            # Get top-k indices
            if k_for_tensor > 0:
                _, indices = torch.topk(flat_scores, k_for_tensor)
                top_indices[name] = indices
        
        return top_indices
    
    def flip_lsb(self, weights, indices):
        """Flip LSBs of parameters at specified indices."""
        perturbed_weights = copy.deepcopy(weights)
        
        for name, idx_tensor in indices.items():
            if name in perturbed_weights:
                # Get the original parameter tensor
                original_tensor = perturbed_weights[name]
                
                # Get flat view for easier indexing
                flat_tensor = original_tensor.view(-1)
                
                # Convert to float32 for bit manipulation
                if flat_tensor.dtype != torch.float32:
                    flat_tensor = flat_tensor.to(torch.float32)
                
                # Convert to binary representation (IEEE 754 for float32)
                binary = torch.frombuffer(flat_tensor.cpu().numpy().tobytes(), dtype=torch.uint8)
                
                # For each index in the current tensor
                for idx in idx_tensor:
                    # Only process indices within bounds
                    if idx < flat_tensor.numel():
                        # Map parameter index to byte index and bit position
                        # Each float32 is 4 bytes
                        byte_idx = (idx.item() * 4) % binary.numel()
                        
                        # Flip the LSB of the mantissa
                        # In IEEE 754, the LSB of the mantissa is the LSB of the last byte
                        binary[byte_idx + 3] ^= 1  # XOR with 1 to flip the LSB
                
                # Convert back to float32 tensor
                perturbed_tensor = torch.frombuffer(binary.numpy().tobytes(), dtype=torch.float32)
                
                # Reshape back to original shape and assign
                perturbed_weights[name] = perturbed_tensor.reshape(original_tensor.shape).to(original_tensor.device)
        
        return perturbed_weights
    
    # Update to the evaluate_accuracy function in fliprl.py
    def evaluate_accuracy(self, eval_data):
        """Evaluate model accuracy on the evaluation dataset."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for item in eval_data:
                # Tokenize input
                input_ids = self.tokenizer(item["input"], return_tensors="pt").to(self.device)
                
                # Get input length and set max_new_tokens appropriately
                input_length = input_ids.input_ids.shape[1]
                
                # Generate output using max_new_tokens instead of max_length
                output = self.model.generate(
                    **input_ids, 
                    max_new_tokens=50,  # Only generate up to 50 new tokens
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated output
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Check if the generated text contains the expected answer
                if item["answer"] in generated_text:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def run_phase1_sensitivity_profiling(self, eval_data):
        """Phase 1: Identify the most sensitive layer."""
        print("Running Phase 1: Layer-wise Sensitivity Profiling...")
        
        baseline_accuracy = self.evaluate_accuracy(eval_data)
        print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
        
        for layer_idx in tqdm(range(self.num_layers), desc="Analyzing layers"):
            # Get original weights for the current layer
            original_weights = self.get_layer_weights(layer_idx)
            
            # Calculate gradients
            gradients = self.calculate_gradients(layer_idx, eval_data)
            
            # Normalize weights and gradients
            norm_weights, norm_gradients = self.normalize_tensors(original_weights, gradients)
            
            # Calculate sensitivity scores
            sensitivity_scores = self.calculate_sensitivity_scores(norm_weights, norm_gradients)
            
            # Determine number of parameters to select
            total_params = sum(tensor.numel() for tensor in original_weights.values())
            k = int(self.selection_rate * total_params)
            
            # Get top-k indices
            subset_indices = self.get_top_k_indices(sensitivity_scores, k)
            
            # Flip LSBs of selected parameters
            perturbed_weights = self.flip_lsb(original_weights, subset_indices)
            
            # Set perturbed weights
            self.set_layer_weights(layer_idx, perturbed_weights)
            
            # Evaluate model with perturbed weights
            perturbed_accuracy = self.evaluate_accuracy(eval_data)
            
            # Restore original weights
            self.set_layer_weights(layer_idx, original_weights)
            
            # Store results
            self.layer_sensitivity_results.append({
                "layer_idx": layer_idx,
                "accuracy": perturbed_accuracy,
                "accuracy_drop": baseline_accuracy - perturbed_accuracy,
                "indices": subset_indices
            })
            
            print(f"Layer {layer_idx}: Accuracy after perturbation: {perturbed_accuracy:.4f} (Drop: {baseline_accuracy - perturbed_accuracy:.4f})")
        
        # Find the most sensitive layer (maximum accuracy drop)
        self.most_sensitive_layer_idx = max(
            self.layer_sensitivity_results, 
            key=lambda x: x["accuracy_drop"]
        )["layer_idx"]
        
        print(f"Most sensitive layer identified: Layer {self.most_sensitive_layer_idx}")
        return self.most_sensitive_layer_idx
    
    def run_phase2_subset_selection(self):
        """Phase 2: Select initial subset of weights for the most sensitive layer."""
        print("Running Phase 2: Weight Subset Selection...")
        
        # Get the indices from the most sensitive layer
        for result in self.layer_sensitivity_results:
            if result["layer_idx"] == self.most_sensitive_layer_idx:
                initial_indices = result["indices"]
                break
        
        print(f"Selected {sum(indices.numel() for indices in initial_indices.values())} parameters for initial RL state")
        return initial_indices
    
    def transition_state(self, state, action):
        """Execute action to transition from current state to next state."""
        next_state = copy.deepcopy(state)
        
        if action == "add":
            # Add a new parameter index to the state
            # Get original weights for the layer
            original_weights = self.get_layer_weights(self.most_sensitive_layer_idx)
            
            # Randomly select a parameter tensor
            tensor_names = list(original_weights.keys())
            name = random.choice(tensor_names)
            
            # Get a random index not already in the state
            tensor = original_weights[name]
            all_indices = set(range(tensor.numel()))
            existing_indices = set(next_state.get(name, torch.tensor([])).tolist())
            candidate_indices = list(all_indices - existing_indices)
            
            if candidate_indices:
                # Add a random new index
                new_idx = random.choice(candidate_indices)
                if name not in next_state:
                    next_state[name] = torch.tensor([new_idx], dtype=torch.long)
                else:
                    # Append to existing tensor
                    next_state[name] = torch.cat([next_state[name], torch.tensor([new_idx], dtype=torch.long)])
        
        elif action == "remove":
            # Remove a parameter index from the state (if not empty)
            non_empty_tensors = [name for name, indices in next_state.items() if indices.numel() > 0]
            
            if non_empty_tensors:
                name = random.choice(non_empty_tensors)
                indices = next_state[name]
                
                if indices.numel() > 0:
                    # Remove a random index
                    idx_to_remove = random.randint(0, indices.numel() - 1)
                    mask = torch.ones(indices.numel(), dtype=torch.bool)
                    mask[idx_to_remove] = False
                    next_state[name] = indices[mask]
        
        elif action == "shift":
            # Shift a parameter index to an adjacent one
            non_empty_tensors = [name for name, indices in next_state.items() if indices.numel() > 0]
            
            if non_empty_tensors:
                name = random.choice(non_empty_tensors)
                indices = next_state[name]
                
                if indices.numel() > 0:
                    # Get original weights for the layer
                    original_weights = self.get_layer_weights(self.most_sensitive_layer_idx)
                    tensor = original_weights[name]
                    
                    # Select a random index to shift
                    idx_to_shift = random.randint(0, indices.numel() - 1)
                    old_idx = indices[idx_to_shift].item()
                    
                    # Determine possible shifts (avoid out of bounds)
                    shifts = []
                    if old_idx > 0:
                        shifts.append(-1)  # Shift left
                    if old_idx < tensor.numel() - 1:
                        shifts.append(1)   # Shift right
                    
                    if shifts:
                        # Apply the shift
                        shift = random.choice(shifts)
                        new_idx = old_idx + shift
                        
                        # Update the index
                        indices[idx_to_shift] = new_idx
                        next_state[name] = indices
        
        return next_state
    
    def select_action_epsilon_greedy(self, state, q_table, epsilon):
        """Select action using epsilon-greedy strategy."""
        # Convert state to a hashable representation
        state_key = self.state_to_key(state)
        
        # With probability epsilon, choose a random action
        if random.random() < epsilon:
            return random.choice(["add", "remove", "shift"])
        
        # Otherwise, choose the best action based on Q-values
        if state_key in q_table:
            action_values = q_table[state_key]
            # Find action with maximum Q-value
            max_value = max(action_values.values())
            # Filter actions that have the maximum value
            best_actions = [a for a, v in action_values.items() if v == max_value]
            return random.choice(best_actions)
        else:
            # If state not in Q-table, choose random action
            return random.choice(["add", "remove", "shift"])
    
    def state_to_key(self, state):
        """Convert a state (dict of tensors) to a hashable key."""
        key_parts = []
        
        # Sort by tensor name for consistency
        for name in sorted(state.keys()):
            indices = state[name]
            # Convert tensor to sorted tuple of indices
            if indices.numel() > 0:
                idx_tuple = tuple(sorted(indices.tolist()))
                key_parts.append(f"{name}:{idx_tuple}")
        
        return "|".join(key_parts)
    
    def extract_optimal_indices(self, state, q_table):
        """Extract the optimal set of indices from the final state."""
        # For simplicity, we're returning the final state as the optimal indices
        # In a more advanced implementation, we might extract based on Q-values
        return state
    
    def count_total_indices(self, indices_dict):
        """Count the total number of indices across all parameter tensors."""
        return sum(indices.numel() for indices in indices_dict.values())
    
    def run_phase3_q_learning(self, eval_data, initial_indices):
        """Phase 3: Q-learning optimization to find critical indices."""
        print("Running Phase 3: Q-Learning Optimization...")
        
        # Initialize Q-table
        q_table = {}  # state_key -> {action -> q_value}
        
        # Initialize current state
        current_state = initial_indices
        
        # Get baseline accuracy
        baseline_accuracy = self.evaluate_accuracy(eval_data)
        
        # Main Q-learning loop
        for generation in tqdm(range(self.rl_generations), desc="RL Training"):
            # Select action using epsilon-greedy
            action = self.select_action_epsilon_greedy(current_state, q_table, self.rl_epsilon)
            
            # Transition to next state
            next_state = self.transition_state(current_state, action)
            
            # Get original weights for the layer
            original_weights = self.get_layer_weights(self.most_sensitive_layer_idx)
            
            # Flip LSBs according to next state
            perturbed_weights = self.flip_lsb(original_weights, next_state)
            
            # Set perturbed weights
            self.set_layer_weights(self.most_sensitive_layer_idx, perturbed_weights)
            
            # Evaluate accuracy
            perturbed_accuracy = self.evaluate_accuracy(eval_data)
            
            # Restore original weights
            self.set_layer_weights(self.most_sensitive_layer_idx, original_weights)
            
            # Calculate reward (maximize accuracy drop while minimizing indices)
            num_indices = self.count_total_indices(next_state)
            if num_indices > 0:
                reward = -(1 - perturbed_accuracy) / num_indices
            else:
                # If no indices, the state doesn't cause any degradation
                reward = 0
            
            # Update Q-value
            current_state_key = self.state_to_key(current_state)
            next_state_key = self.state_to_key(next_state)
            
            # Initialize state in Q-table if not present
            if current_state_key not in q_table:
                q_table[current_state_key] = {"add": 0, "remove": 0, "shift": 0}
            
            # Get current Q-value
            current_q = q_table[current_state_key][action]
            
            # Get max Q-value for next state
            if next_state_key in q_table:
                max_next_q = max(q_table[next_state_key].values())
            else:
                # Initialize with zeros if not in Q-table
                q_table[next_state_key] = {"add": 0, "remove": 0, "shift": 0}
                max_next_q = 0
            
            # Update Q-value using Bellman equation
            new_q = (1 - self.rl_alpha) * current_q + self.rl_alpha * (reward + self.rl_gamma * max_next_q)
            q_table[current_state_key][action] = new_q
            
            # Move to next state
            current_state = next_state
            
            # Print progress
            if (generation + 1) % (self.rl_generations // 5) == 0 or generation == 0:
                num_indices = self.count_total_indices(current_state)
                print(f"Generation {generation + 1}/{self.rl_generations}: "
                      f"Accuracy: {perturbed_accuracy:.4f}, Indices: {num_indices}, "
                      f"Reward: {reward:.4f}")
        
        # Extract optimal indices from final state
        self.critical_indices = self.extract_optimal_indices(current_state, q_table)
        
        # Final evaluation with critical indices
        original_weights = self.get_layer_weights(self.most_sensitive_layer_idx)
        final_perturbed_weights = self.flip_lsb(original_weights, self.critical_indices)
        self.set_layer_weights(self.most_sensitive_layer_idx, final_perturbed_weights)
        self.final_accuracy = self.evaluate_accuracy(eval_data)
        self.set_layer_weights(self.most_sensitive_layer_idx, original_weights)
        
        # Report results
        num_critical_indices = self.count_total_indices(self.critical_indices)
        print(f"\nResults:")
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"Final Accuracy: {self.final_accuracy:.4f}")
        print(f"Number of Critical Bit-Flips: {num_critical_indices}")
        
        return self.critical_indices, self.final_accuracy
    
    def run_random_baseline(self, eval_data, num_random_flips=10000):
        """Run random LSB flips baseline for comparison."""
        print(f"Running Random LSB Flips Baseline (N={num_random_flips})...")
        
        # Get original weights for the most sensitive layer
        original_weights = self.get_layer_weights(self.most_sensitive_layer_idx)
        
        # Create random indices
        random_indices = {}
        for name, tensor in original_weights.items():
            num_params = tensor.numel()
            # Ensure we don't request more indices than we have parameters
            k = min(num_random_flips, num_params)
            if k > 0:
                random_idx = torch.randperm(num_params)[:k]
                random_indices[name] = random_idx
        
        # Flip LSBs according to random indices
        perturbed_weights = self.flip_lsb(original_weights, random_indices)
        
        # Set perturbed weights
        self.set_layer_weights(self.most_sensitive_layer_idx, perturbed_weights)
        
        # Evaluate accuracy
        random_accuracy = self.evaluate_accuracy(eval_data)
        
        # Restore original weights
        self.set_layer_weights(self.most_sensitive_layer_idx, original_weights)
        
        # Report results
        baseline_accuracy = self.evaluate_accuracy(eval_data)
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"Random Flips Accuracy: {random_accuracy:.4f}")
        print(f"Number of Random Flips: {num_random_flips}")
        
        return random_accuracy
    
    def run_complete_pipeline(self, eval_data):
        """Run the complete FlipRL pipeline."""
        start_time = time.time()
        
        # Phase 1: Layer-wise Sensitivity Profiling
        self.run_phase1_sensitivity_profiling(eval_data)
        
        # Phase 2: Weight Subset Selection
        initial_indices = self.run_phase2_subset_selection()
        
        # Phase 3: Q-Learning Optimization
        self.run_phase3_q_learning(eval_data, initial_indices)
        
        # Run random baseline for comparison
        self.run_random_baseline(eval_data)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        
        return {
            "most_sensitive_layer": self.most_sensitive_layer_idx,
            "critical_indices": self.critical_indices,
            "final_accuracy": self.final_accuracy,
            "execution_time": elapsed_time
        }

# Function to prepare MMLU dataset
def prepare_mmlu_subset(num_examples=50):
    """Prepare a subset of the MMLU dataset for evaluation."""
    # Load a subset of MMLU
    try:
        mmlu = load_dataset("cais/mmlu", "high_school_mathematics", split="validation")
        
        # Take a small subset for demonstration
        mmlu_subset = mmlu.select(range(min(num_examples, len(mmlu))))
        
        # Format data for our evaluation
        eval_data = []
        for item in mmlu_subset:
            # Create prompt with question and choices
            prompt = f"Question: {item['question']}\n"
            prompt += f"A. {item['choices'][0]}\n"
            prompt += f"B. {item['choices'][1]}\n"
            prompt += f"C. {item['choices'][2]}\n"
            prompt += f"D. {item['choices'][3]}\n"
            prompt += "Answer: "
            
            # Get correct answer
            correct_choice = ["A", "B", "C", "D"][item["answer"]]
            
            eval_data.append({
                "input": prompt,
                "output": correct_choice,
                "answer": correct_choice
            })
        
        return eval_data
    
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        # Create synthetic data for demonstration
        print("Creating synthetic evaluation data instead")
        eval_data = []
        for i in range(num_examples):
            choices = ["Paris", "London", "Berlin", "Madrid"]
            correct_idx = i % 4
            
            prompt = f"Question: What is the capital of Country {i}?\n"
            prompt += f"A. {choices[0]}\n"
            prompt += f"B. {choices[1]}\n"
            prompt += f"C. {choices[2]}\n"
            prompt += f"D. {choices[3]}\n"
            prompt += "Answer: "
            
            correct_choice = ["A", "B", "C", "D"][correct_idx]
            
            eval_data.append({
                "input": prompt,
                "output": correct_choice,
                "answer": correct_choice
            })
        
        return eval_data

# Main execution
def main():
    # Initialize FlipRL framework
    fliprl = FlipRL(
        model_name="gpt2-large",  # Using GPT-2 Large (774M)
        alpha=0.5,                # Equal weight for gradient and magnitude
        selection_rate=0.001,     # 0.1% of parameters
        rl_generations=10,        # Number of Q-learning generations
    )
    
    # Prepare evaluation data
    eval_data = prepare_mmlu_subset(num_examples=20)
    
    # Run the complete pipeline
    results = fliprl.run_complete_pipeline(eval_data)
    
    # Print summary
    print("\nFlipRL Execution Summary:")
    print(f"Model: GPT-2 Large")
    print(f"Most sensitive layer: {results['most_sensitive_layer']}")
    print(f"Number of critical bit-flips: {sum(indices.numel() for indices in results['critical_indices'].values())}")
    print(f"Final accuracy after attack: {results['final_accuracy']:.4f}")
    print(f"Total execution time: {results['execution_time']:.2f} seconds")

if __name__ == "__main__":
    main()