import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from Models import SimpleNN, BayesianNN
import os
from typing import Dict, List, Tuple, Union, Optional, NamedTuple
import logging
from pathlib import Path
from decorators import log_and_handle_errors, LogLevel
from tqdm import tqdm
from graph_plotting import plot_model_analysis
from torch.utils.data import DataLoader
from calibration_plotting import (
    compute_ece, compute_adaptive_ece, compute_conditional_correlation,
    plot_gap_style_calibration_curve, brier_score
)

# Set up logging
logger = logging.getLogger(__name__)

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def get_model_predictions(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and convert to numpy arrays."""
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y_pred = model(X).cpu().numpy()  # Shape: (n_samples, n_classes)
        y_pred_binary = np.argmax(y_pred, axis=1)  # Get predicted class (0-9)
        return y_pred, y_pred_binary

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for model predictions."""
    # Ensure labels are in correct range (0-9)
    y_true = y_true.astype(int)
    y_pred_binary = y_pred_binary.astype(int)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, average='weighted'),
        "f1_score": f1_score(y_true, y_pred_binary, average='weighted'),
        "ece": compute_ece(y_pred_proba, y_true),
        "adaptive_ece": compute_adaptive_ece(y_pred_proba, y_true),
        "conditional_correlation": compute_conditional_correlation(y_pred_proba, y_true),
        "brier_score": brier_score(y_pred_proba, y_true)
    }

@log_and_handle_errors(log_level=LogLevel.INFO)
def train_simplenn(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
) -> nn.Module:
    """Train a simple neural network using Adam optimizer and CrossEntropy loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    pbar = tqdm(range(epochs), desc="Training SimpleNN", unit="epoch")
    for epoch in pbar:
        total_loss = 0
        for features, labels in train_loader:
            # Transfer data to GPU in batches and reshape features
            features = features.view(features.size(0), -1).to(device, non_blocking=True)  # Flatten images
            labels = labels.to(device, non_blocking=True)  # Labels are already class indices
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'}, refresh=True)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate model performance and plot calibration curve."""
    y_pred, y_pred_binary = get_model_predictions(model, X_test, device)
    y_true = y_test.numpy().flatten()  # Already in correct range (0-9)
    
    # Only plot calibration curve for SimpleNN
    if model_name == "SimpleNN":
        plot_gap_style_calibration_curve(y_pred, y_true, model_name=model_name, num_bins=15)
    
    return compute_metrics(y_true, y_pred_binary, y_pred)

@log_and_handle_errors(log_level=LogLevel.INFO)
def sgld_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    noise_std: float,
    device: torch.device,
    temperature: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """Sample from the posterior using Stochastic Gradient Langevin Dynamics (SGLD).
    Simplified version focusing on core Markov chain sampling."""
    model = model.to(device)
    posterior_samples = []
    burnin_epochs = 700  # Fixed burn-in period
    target_samples = 10  # We want exactly 10 models
    dataset_size = len(train_loader.dataset)
    
    logger.info(f"Starting SGLD sampling: {epochs} total epochs, {burnin_epochs} burn-in epochs")
    logger.info(f"Temperature: {temperature}, Learning rate: {lr}, Noise std: {noise_std}")
    
    # Initialize parameters from prior
    for param in model.parameters():
        param.data = torch.randn_like(param) * model.prior_std
    
    # Convert constants to tensors on the correct device
    temperature_tensor = torch.tensor(temperature, device=device)
    noise_std_tensor = torch.tensor(noise_std, device=device)
    two_tensor = torch.tensor(2.0, device=device)
    
    model.train()
    pbar = tqdm(range(epochs), desc="Sampling from Posterior", unit="epoch")
    
    for epoch in pbar:
        try:
            for batch_features, batch_labels in train_loader:
                # Flatten features and move to device
                batch_features = batch_features.view(batch_features.size(0), -1).to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(batch_features)
                
                # Compute negative log likelihood with temperature scaling
                nll = F.cross_entropy(outputs, batch_labels, reduction='mean')
                
                # Compute prior (L2 regularization)
                log_prior = model.log_prior() / dataset_size
                
                # Compute loss (negative log posterior)
                loss = nll - (temperature * log_prior)
                
                # Compute gradients
                loss.backward()
                
                # SGLD update step
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            # Calculate noise scale
                            noise_scale = torch.sqrt(two_tensor * lr / temperature_tensor)
                            
                            # Generate noise
                            noise = torch.randn_like(param) * noise_scale * noise_std_tensor
                            
                            # Update parameters
                            param.data.add_(-lr * param.grad)
                            param.data.add_(noise)
                            
                            # Zero gradients
                            param.grad.zero_()
            
            # Store samples after burn-in
            if epoch >= burnin_epochs:
                # Sample every 30 epochs after burn-in until we have 10 models
                if (epoch - burnin_epochs) % 30 == 0 and len(posterior_samples) < target_samples:
                    sample = {
                        name: param.data.clone().cpu()
                        for name, param in model.named_parameters()
                    }
                    posterior_samples.append(sample)
                    logger.info(f"Collected sample {len(posterior_samples)}/{target_samples} at epoch {epoch}")
                    
                    # If we have all 10 models, we can stop
                    if len(posterior_samples) == target_samples:
                        logger.info("Collected all required samples, stopping early")
                        break
            
            # Update progress bar
            pbar.set_postfix({
                'samples': f"{len(posterior_samples)}/{target_samples}"
            }, refresh=True)
                
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            continue
    
    logger.info(f"Sampling completed. Collected {len(posterior_samples)} samples after burn-in")
    return posterior_samples

@log_and_handle_errors(log_level=LogLevel.INFO)
def create_bayesian_models(
    posterior_samples: List[Dict[str, torch.Tensor]],
    input_dim: int,
    num_samples: int,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> List[nn.Module]:
    """Create individual Bayesian models from posterior parameter samples."""
    models = []
    for sample in tqdm(posterior_samples[:num_samples], desc="Creating Bayesian models", unit="model"):
        model = BayesianNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        ).to(device)  # Move model to device immediately after creation
        
        # Copy parameters from sample to model, ensuring they're on the correct device
        for name, param in model.named_parameters():
            if name in sample:
                param.data.copy_(sample[name].to(device))  # Ensure sample is on the correct device
        models.append(model)
    return models

@log_and_handle_errors(log_level=LogLevel.INFO)
def evaluate_ensemble(
    models: List[nn.Module],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate ensemble performance by averaging predictions from multiple models."""
    all_predictions = []
    
    with torch.no_grad():
        X_test = X_test.to(device)
        for model in tqdm(models, desc="Evaluating ensemble models", unit="model"):
            model.eval()
            pred = model(X_test).cpu().numpy()
            all_predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    ensemble_pred_binary = np.argmax(ensemble_pred, axis=1)
    y_true = y_test.numpy().flatten()  # Already in correct range (0-9)
    
    # Plot calibration curve for ensemble
    plot_gap_style_calibration_curve(ensemble_pred, y_true, model_name="BayesianNN_Ensemble", num_bins=15)
    
    return compute_metrics(y_true, ensemble_pred_binary, ensemble_pred)

@log_and_handle_errors(log_level=LogLevel.INFO)
def save_results(results: Dict[str, Dict[str, float]], filename: str) -> None:
    """Save model evaluation results to a CSV file with improved formatting."""
    try:
        # Get the results directory path
        current_dir = Path(__file__).parent
        results_dir = current_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Create the full file path
        file_path = results_dir / filename
        
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # Round all numeric values to 4 decimal places
        df = df.round(4)
        
        # Add a blank row between models for better readability
        df_with_spacing = pd.DataFrame()
        for idx, row in df.iterrows():
            df_with_spacing = pd.concat([df_with_spacing, pd.DataFrame([row])])
            df_with_spacing = pd.concat([df_with_spacing, pd.DataFrame([pd.Series()])])
        
        # Save with improved formatting
        with open(file_path, 'w', newline='') as f:
            # Write header
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Write the DataFrame
            df_with_spacing.to_csv(f, sep=',', index=True)
            
            # Add a footer with timestamp
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Results saved to {file_path}")
        
        # Also save a clean version without formatting for programmatic use
        clean_file_path = results_dir / f"clean_{filename}"
        df.to_csv(clean_file_path)
        logger.info(f"Clean results saved to {clean_file_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def compute_calibration_metrics(y_pred_proba: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> Dict[str, float]:
    """Compute calibration metrics including ECE and confidence distribution."""
    # Get predicted probabilities and true labels
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    correct = (predictions == y_true).astype(float)
    
    # Compute bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    # Initialize arrays for bin statistics
    bin_counts = np.zeros(num_bins)
    bin_correct = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    
    # Compute bin statistics
    for i in range(num_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_correct[i] = np.sum(correct[mask])
            bin_conf[i] = np.mean(confidences[mask])
    
    # Compute ECE
    ece = np.sum(np.abs(bin_correct - bin_conf) * bin_counts) / np.sum(bin_counts)
    
    # Compute confidence distribution statistics
    conf_mean = np.mean(confidences)
    conf_std = np.std(confidences)
    conf_range = np.min(confidences), np.max(confidences)
    
    return {
        'ece': ece,
        'conf_mean': conf_mean,
        'conf_std': conf_std,
        'conf_range': conf_range,
        'bin_counts': bin_counts,
        'bin_correct': bin_correct,
        'bin_conf': bin_conf,
        'bin_edges': bin_edges
    }
