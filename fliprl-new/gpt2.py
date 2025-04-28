import torch
import os
import logging
from fliprl import FlipRL, prepare_mmlu_subset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fliprl_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FlipRL")

def main():
    logger.info("Starting FlipRL execution")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # FlipRL parameters
    params = {
        "model_name": "gpt2-large",
        "alpha": 0.5,              # Sensitivity mixing coefficient
        "selection_rate": 0.001,   # 0.1% of parameters
        "device": device,
        "rl_learning_rate": 0.1,   # Q-learning learning rate
        "rl_discount_factor": 0.9, # Discount factor for future rewards
        "rl_exploration_rate": 0.1, # Exploration rate for epsilon-greedy
        "rl_generations": 10       # Number of Q-learning generations
    }
    
    logger.info(f"FlipRL Parameters: {params}")
    
    try:
        # Initialize FlipRL
        fliprl = FlipRL(**params)
        logger.info("FlipRL initialized successfully")
        
        # Prepare evaluation data
        logger.info("Preparing MMLU evaluation data")
        eval_data = prepare_mmlu_subset(num_examples=20)
        logger.info(f"Prepared {len(eval_data)} evaluation examples")
        
        # Run the complete pipeline
        logger.info("Starting FlipRL pipeline execution")
        results = fliprl.run_complete_pipeline(eval_data)
        
        # Log and print summary
        logger.info("FlipRL execution completed successfully")
        logger.info(f"Most sensitive layer: {results['most_sensitive_layer']}")
        logger.info(f"Number of critical bit-flips: {sum(indices.numel() for indices in results['critical_indices'].values())}")
        logger.info(f"Final accuracy after attack: {results['final_accuracy']:.4f}")
        logger.info(f"Total execution time: {results['execution_time']:.2f} seconds")
        
        # Save results
        torch.save(results, "fliprl_results.pt")
        logger.info("Results saved to fliprl_results.pt")
        
    except Exception as e:
        logger.error(f"Error during FlipRL execution: {e}", exc_info=True)
    
    logger.info("FlipRL execution finished")

if __name__ == "__main__":
    main()