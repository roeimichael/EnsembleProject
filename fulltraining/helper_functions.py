import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from Models import SimpleNN, BayesianNN
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from decorators import log_and_handle_errors, LogLevel
from tqdm import tqdm
from graph_plotting import plot_model_analysis, plot_calibration_curve
from torch.utils.data import DataLoader
from calibration_plotting import (
    compute_ece, compute_adaptive_ece, compute_conditional_correlation,
    plot_gap_style_calibration_curve, brier_score
)

logger = logging.getLogger(__name__)

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def get_model_predictions(model: nn.Module, X: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y_pred = model(X).cpu().numpy()
        y_pred_binary = np.argmax(y_pred, axis=1)
        return y_pred, y_pred_binary

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
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

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def load_simplenn(input_size: int, hidden_dims: List[int], device: torch.device) -> nn.Module:
    """Load a pre-trained SimpleNN model from the saved directory."""
    try:
        current_dir = Path(__file__).parent
        model_dir = current_dir / "saved_models"
        model_path = model_dir / "simplenn.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        logger.info(f"Loading SimpleNN from {model_path}")
        model = SimpleNN(input_size, hidden_dims)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("SimpleNN loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SimpleNN: {e}")
        raise

@log_and_handle_errors(log_level=LogLevel.INFO)
def train_simplenn(model: nn.Module, train_loader: torch.utils.data.DataLoader, epochs: int, lr: float, device: torch.device, save_model: bool = True) -> nn.Module:
    """Train a SimpleNN model or save it if training is complete."""
    try:
        current_dir = Path(__file__).parent
        model_dir = current_dir / "saved_models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "simplenn.pth"
        
        if model_path.exists():
            logger.info(f"Found saved model at {model_path}, skipping training")
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model
            
        # Original training code
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        scaler = torch.amp.GradScaler()
        model.train()
        pbar = tqdm(range(epochs), desc="Training SimpleNN", unit="epoch")
        for epoch in pbar:
            total_loss = 0
            for features, labels in train_loader:
                features = features.view(features.size(0), -1).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'}, refresh=True)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        if save_model:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved trained model to {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error in train_simplenn: {e}")
        raise

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device, model_name: str = "Model") -> Dict[str, float]:
    y_pred, y_pred_binary = get_model_predictions(model, X_test, device)
    y_true = y_test.numpy().flatten()
    if model_name == "SimpleNN":
        plot_calibration_curve(model_name, y_pred, y_true)
    return compute_metrics(y_true, y_pred_binary, y_pred)

@log_and_handle_errors(log_level=LogLevel.INFO)
def sgld_training(model: nn.Module, train_loader: torch.utils.data.DataLoader, epochs: int, lr: float, noise_std: float, device: torch.device, temperature: float = 1.0) -> List[Dict[str, torch.Tensor]]:
    model = model.to(device)
    posterior_samples = []
    target_samples = 10
    dataset_size = len(train_loader.dataset)
    
    # Calculate adaptive sampling points
    training_phase = int(epochs * 0.75)  # First 75% for pure training
    sampling_phase = epochs - training_phase  # Last 25% for sampling
    sample_interval = sampling_phase // target_samples
    
    logger.info(f"SGLD Training Strategy:")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Training phase: epochs 1-{training_phase} (no sampling)")
    logger.info(f"Sampling phase: epochs {training_phase+1}-{epochs}")
    logger.info(f"Will collect {target_samples} samples every {sample_interval} epochs during sampling phase")
    
    # Initialize model parameters
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.randn_like(param) * (model.prior_std * 0.5)
    
    # Setup training parameters
    temperature_tensor = torch.tensor(temperature, device=device)
    noise_std_tensor = torch.tensor(noise_std, device=device)
    two_tensor = torch.tensor(2.0, device=device)
    
    model.train()
    pbar = tqdm(range(epochs), desc="SGLD Training", unit="epoch")
    
    # Initialize metrics tracking
    best_loss = float('inf')
    patience = 0
    max_patience = 50
    moving_avg_acc = 0.0
    alpha = 0.1  # For exponential moving average
    
    for epoch in pbar:
        try:
            total_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            # Training loop
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.view(batch_features.size(0), -1).to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Compute accuracy for this batch
                epoch_total += batch_labels.size(0)
                epoch_correct += (predicted == batch_labels).sum().item()
                
                # Compute loss
                nll = F.cross_entropy(outputs, batch_labels, reduction='mean')
                log_prior = model.log_prior()
                batch_size = batch_features.size(0)
                prior_scale = batch_size / dataset_size
                loss = nll + (temperature / dataset_size) * log_prior * prior_scale
                total_loss += loss.item()
                
                # Backward pass and SGLD update
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
                            noise_scale = torch.sqrt(two_tensor * lr)
                            if epoch < training_phase:
                                noise_scale *= 0.5  # Reduced noise during training phase
                            elif patience > max_patience:
                                noise_scale *= 2.0  # Increased noise when stuck
                            noise = torch.randn_like(param) * noise_scale * noise_std_tensor
                            param.data.add_(-lr * param.grad)
                            param.data.add_(noise)
                            param.grad = None
            
            # Compute epoch metrics
            avg_loss = total_loss / len(train_loader)
            epoch_accuracy = epoch_correct / epoch_total
            moving_avg_acc = alpha * epoch_accuracy + (1 - alpha) * moving_avg_acc
            
            # Update best loss and patience
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
            
            # Collect samples during sampling phase
            if epoch >= training_phase and (epoch - training_phase) % sample_interval == 0 and len(posterior_samples) < target_samples:
                if moving_avg_acc > 0.35:  # Minimum accuracy threshold
                    sample = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
                    posterior_samples.append(sample)
                    logger.info(f"Collected sample {len(posterior_samples)}/{target_samples} at epoch {epoch}, accuracy: {epoch_accuracy:.3f}")
                else:
                    logger.warning(f"Skipped sample at epoch {epoch} due to low accuracy: {epoch_accuracy:.3f}")
            
            # Update progress bar
            pbar.set_postfix({
                'phase': 'sampling' if epoch >= training_phase else 'training',
                'samples': f"{len(posterior_samples)}/{target_samples}",
                'loss': f"{avg_loss:.4f}",
                'acc': f"{epoch_accuracy:.3f}",
                'patience': patience
            }, refresh=True)
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            continue
    
    if len(posterior_samples) < target_samples:
        logger.warning(f"Only collected {len(posterior_samples)} samples out of {target_samples} desired samples")
    
    logger.info(f"SGLD Training completed:")
    logger.info(f"Final loss: {avg_loss:.4f}")
    logger.info(f"Final accuracy: {epoch_accuracy:.3f}")
    logger.info(f"Collected samples: {len(posterior_samples)}/{target_samples}")
    
    return posterior_samples

@log_and_handle_errors(log_level=LogLevel.INFO)
def create_bayesian_models(posterior_samples: List[Dict[str, torch.Tensor]], input_dim: int, num_samples: int, hidden_dims: Tuple[int, ...] = (64, 32), dropout_rate: float = 0.2, use_batch_norm: bool = True, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[nn.Module]:
    models = []
    for sample in tqdm(posterior_samples[:num_samples], desc="Creating Bayesian models", unit="model"):
        model = BayesianNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm).to(device)
        for name, param in model.named_parameters():
            if name in sample:
                param.data.copy_(sample[name].to(device))
        models.append(model)
    return models

@log_and_handle_errors(log_level=LogLevel.INFO)
def evaluate_ensemble(models: List[nn.Module], X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device) -> Dict[str, float]:
    # Ensure consistent dimensions
    n_samples = min(len(X_test), len(y_test))
    X_test = X_test[:n_samples]
    y_test = y_test[:n_samples]
    
    logger.info(f"Evaluating ensemble with {n_samples} samples")
    
    all_probs = []
    pbar = tqdm(models, desc="Evaluating ensemble models", unit="model")
    with torch.no_grad():
        for model in pbar:
            model.eval()
            outputs = model(X_test.to(device))
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
    
    ensemble_pred = torch.stack(all_probs).mean(dim=0)
    y_pred = ensemble_pred.argmax(dim=1).numpy()
    y_true = y_test.cpu().numpy()
    
    metrics = compute_metrics(y_true, y_pred, ensemble_pred.numpy())
    
    # Use plot_calibration_curve from graph_plotting
    plot_calibration_curve("BayesianNN_Ensemble", ensemble_pred.numpy(), y_true)
    
    print("\nConfidence Distribution Summary for BayesianNN_Ensemble:")
    max_probs = ensemble_pred.max(dim=1)[0].numpy()
    print(f"Range: {max_probs.min():.3f} to {max_probs.max():.3f}")
    print(f"Mean: {max_probs.mean():.3f}")
    print(f"Std: {max_probs.std():.3f}")
    
    return metrics

@log_and_handle_errors(log_level=LogLevel.INFO)
def save_results(results: Dict[str, Dict[str, float]], filename: str) -> None:
    try:
        current_dir = Path(__file__).parent
        results_dir = current_dir / "results"
        results_dir.mkdir(exist_ok=True)
        file_path = results_dir / filename
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.round(4)
        df_with_spacing = pd.DataFrame()
        for idx, row in df.iterrows():
            df_with_spacing = pd.concat([df_with_spacing, pd.DataFrame([row])])
            df_with_spacing = pd.concat([df_with_spacing, pd.DataFrame([pd.Series()])])
        with open(file_path, 'w', newline='') as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            df_with_spacing.to_csv(f, sep=',', index=True)
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Results saved to {file_path}")
        clean_file_path = results_dir / f"clean_{filename}"
        df.to_csv(clean_file_path)
        logger.info(f"Clean results saved to {clean_file_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def compute_calibration_metrics(y_pred_proba: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> Dict[str, float]:
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_counts = np.zeros(num_bins)
    bin_correct = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_correct[i] = np.sum(correct[mask])
            bin_conf[i] = np.mean(confidences[mask])
    ece = np.sum(np.abs(bin_correct - bin_conf) * bin_counts) / np.sum(bin_counts)
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