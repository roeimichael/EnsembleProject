import numpy as np
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BasicTraining.helper_functions import load_config, load_dataset, prepare_data
from BasicTraining.sgld import SGLDTrainer
from BasicTraining.model_evaluation import analyze_ensemble

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

def main():
    config = load_config('BasicTraining/config.yaml')
    setup_logging(config['paths']['logs'])
    
    X, y = load_dataset(config['data']['name'])
    X_train, X_val, y_train, y_val = prepare_data(
        X, y,
        train_size=config['data']['train_size'],
        random_state=config['data']['random_state']
    )
    
    trainer = SGLDTrainer(
        input_dim=X.shape[1],
        num_classes=len(np.unique(y)),
        prior_std=config['sgld']['prior_std'],
        temperature=config['sgld']['temperature'],
        device=config['training']['device']
    )
    
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        num_epochs=config['sgld']['epochs'],
        batch_size=config['data']['batch_size'],
        learning_rate=config['sgld']['learning_rate'],
        sampling_percentage=config['sgld']['sampling_percentage'],
        num_ensemble_models=config['sgld']['num_ensemble_models']
    )
    
    samples, history = trainer.get_models()
    
    analyze_ensemble(
        trainer.model,
        samples,
        X_val,
        y_val,
        trainer.device,
        history,
        save_dir=Path(config['paths']['logs'])
    )

if __name__ == "__main__":
    main() 