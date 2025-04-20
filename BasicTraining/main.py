import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import yaml
from helper_functions import load_dataset, prepare_data, plot_training_history, plot_calibration_curve
from simple_nn import SimpleNN, train_model, evaluate_model
from sgld import LinearModel, SGLDTrainer, plot_training_comparison, plot_predictions_comparison

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir):
    """Set up logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Load configuration
    config = load_config('BasicTraining/config.yaml')
    
    # Setup logging
    setup_logging(config['paths']['logs'])
    
    # Load and prepare data
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = prepare_data(X, y, config['data']['train_size'])
    
    # Initialize SGLD trainer
    trainer = SGLDTrainer(
        input_dim=X.shape[1],
        num_classes=len(np.unique(y)),
        prior_std=config['sgld']['prior_std'],
        temperature=config['sgld']['temperature'],
        device=config['training']['device']
    )
    
    # Train model
    logging.info("Starting training...")
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        num_epochs=config['sgld']['epochs'],
        batch_size=config['data']['batch_size'],
        learning_rate=config['sgld']['learning_rate']
    )
    
    # Analyze models and save results
    logging.info("Analyzing models...")
    results = trainer.analyze_models(
        X_val, y_val,
        save_path=Path(config['paths']['logs']) / 'model_analysis.csv'
    )
    logging.info(f"Model analysis results:\n{results}")
    
    # Plot training comparison
    plot_training_comparison(
        trainer.history,
        save_path=Path(config['paths']['plots']) / 'training_comparison.png'
    )
    
    # Plot predictions comparison
    plot_predictions_comparison(
        trainer, X_val, y_val,
        save_path=Path(config['paths']['plots']) / 'predictions_comparison.png'
    )
    
    logging.info("Training and analysis complete!")

if __name__ == "__main__":
    main() 