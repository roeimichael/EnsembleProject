import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from Models import SimpleNN, BayesianNN
from helper_functions import (
    train_simplenn, 
    sgld_training, 
    evaluate_model, 
    evaluate_ensemble, 
    plot_model_analysis,
    load_simplenn
)
from model_manager import save_model, load_model, save_sgld_samples, load_sgld_samples
from config_manager import ConfigManager

def load_and_prepare_data(config: ConfigManager):
    """Load and prepare the Fashion-MNIST dataset."""
    data_config = config.get_data_config()
    transform_config = data_config['transform']
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((transform_config['normalize_mean'],), 
                           (transform_config['normalize_std'],))
    ])
    
    # Load full dataset
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Take a smaller subset of the data
    train_size = int(data_config['train_size'] * len(trainset))
    test_size = int(data_config['train_size'] * len(testset))
    
    trainset, _ = torch.utils.data.random_split(trainset, [train_size, len(trainset) - train_size])
    testset, _ = torch.utils.data.random_split(testset, [test_size, len(testset) - test_size])
    
    train_loader = DataLoader(
        trainset, 
        batch_size=data_config['batch_size'], 
        shuffle=True, 
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    # Prepare test data and flatten images
    X_test = torch.stack([x for x, _ in testset])
    X_test = X_test.view(X_test.size(0), -1)  # Flatten images to (n_samples, 784)
    y_test = torch.tensor([y for _, y in testset])
    
    print(f"Using {train_size} training samples and {test_size} test samples")
    return train_loader, X_test, y_test

def main():
    # Load configuration
    config = ConfigManager()
    
    torch.manual_seed(config.get_random_seed())
    np.random.seed(config.get_random_seed())
    device = torch.device(config.get_device() if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, X_test, y_test = load_and_prepare_data(config)
    model_config = config.get_model_config()
    input_size = model_config['input_size']
    hidden_dims = tuple(model_config['hidden_dims'])

    print("\nLoading SimpleNN model from saved file...")
    try:
        simplenn_model = load_simplenn(input_size, hidden_dims, device)
        print("Successfully loaded SimpleNN model")
    except FileNotFoundError:
        print("No saved model found. Training new SimpleNN model...")
        simplenn_model = SimpleNN(
            input_size, 
            hidden_dims,
            dropout_rate=model_config['dropout_rate'],
            use_batch_norm=model_config['use_batch_norm']
        ).to(device)
        
        simplenn_config = config.get_simplenn_config()
        train_simplenn(
            simplenn_model, 
            train_loader, 
            epochs=simplenn_config['epochs'],
            lr=simplenn_config['learning_rate'],
            device=device,
            save_model=True
        )
    
    # Train SGLD models (sample from posterior)
    print("\nSampling from posterior using SGLD...")
    model = BayesianNN(
        input_size, 
        hidden_dims,
        dropout_rate=model_config['dropout_rate'],
        use_batch_norm=model_config['use_batch_norm'],
        prior_std=model_config['prior_std']
    ).to(device)
    
    sgld_config = config.get_sgld_config()
    sgld_samples = sgld_training(
        model=model,
        train_loader=train_loader,
        epochs=sgld_config['epochs'],
        lr=sgld_config['learning_rate'],
        noise_std=sgld_config['noise_std'],
        device=device,
        temperature=sgld_config['temperature']
    )
    save_sgld_samples(sgld_samples, "sgld")
    
    # Create models from samples
    sgld_models = []
    for i, sample in enumerate(sgld_samples):
        model = BayesianNN(
            input_size, 
            hidden_dims,
            dropout_rate=model_config['dropout_rate'],
            use_batch_norm=model_config['use_batch_norm'],
            prior_std=model_config['prior_std']
        ).to(device)
        for name, param in model.named_parameters():
            if name in sample:
                param.data.copy_(sample[name].to(device))
        sgld_models.append(model)
    
    # Evaluate SimpleNN model
    print("\nEvaluating SimpleNN model...")
    metrics = evaluate_model(simplenn_model, X_test, y_test, device, "SimpleNN")
    print("\nMetrics for SimpleNN:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate SGLD models
    print("\nEvaluating SGLD models...")
    for i, model in enumerate(sgld_models):
        print(f"\nEvaluating SGLD model {i+1}/{len(sgld_models)}")
        metrics = evaluate_model(model, X_test, y_test, device, f"SGLD_{i+1}")
        print(f"Metrics for SGLD_{i+1}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Evaluate ensemble (SimpleNN + SGLD samples)
    print("\nEvaluating ensemble...")
    ensemble_metrics = evaluate_ensemble([simplenn_model] + sgld_models, X_test, y_test, device)
    print("\nEnsemble Metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot model analysis
    print("\nGenerating model analysis plots...")
    plot_model_analysis(
        [simplenn_model] + sgld_models, 
        X_test, 
        y_test, 
        lr=simplenn_config['learning_rate'],
        noise_std=sgld_config['noise_std'],
        device=device
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        input("Press Enter to exit...")  # This will keep the window open 