import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from torch import nn, optim

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibrating model confidence."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits, labels, max_iter=50, lr=0.01):
        """Tune the temperature parameter on validation data."""
        self.to(logits.device)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        return self

def compute_true_class_matrix(all_probs, y):
    P = torch.zeros((len(y), len(all_probs)))
    for n in range(len(y)):
        for m in range(len(all_probs)):
            P[n, m] = all_probs[m, n, y[n]]
    return P

def compute_certainty(P):
    return P.mean(dim=1)


def plot_training_history(history, sampling_epochs, save_path):
    """Plot training and validation metrics over time with sampling points marked."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', color='blue')
    plt.plot(history['val_loss'], label='Validation', color='red')
    
    # Mark sampling points
    for epoch in sampling_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
        plt.scatter(epoch, history['val_loss'][epoch], color='red', marker='x', s=100)
    
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', color='blue')
    plt.plot(history['val_acc'], label='Validation', color='red')
    
    # Mark sampling points
    for epoch in sampling_epochs:
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
        plt.scatter(epoch, history['val_acc'][epoch], color='red', marker='x', s=100)
    
    plt.title('Accuracy over time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(y_true, y_prob, save_path, n_bins=10):
    """Plot calibration curve as a bar chart with accuracy per confidence bin."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate confidence values and prediction accuracy
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    # Define bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_widths = bin_uppers - bin_lowers
    
    # Calculate accuracy and samples per bin
    acc_in_bins = []
    count_in_bins = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        if np.any(in_bin):
            acc_in_bin = np.mean(accuracies[in_bin])
            count_in_bin = np.sum(in_bin)
        else:
            acc_in_bin = 0
            count_in_bin = 0
        
        acc_in_bins.append(acc_in_bin)
        count_in_bins.append(count_in_bin)
    
    # Plot the bars
    bars = ax.bar(bin_centers, acc_in_bins, width=bin_widths * 0.9, 
                 color='blue', edgecolor='black', alpha=0.7, label='Outputs')
    
    # Plot the diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Fill the gap between bars and diagonal line
    for i, (x, y, width) in enumerate(zip(bin_centers, acc_in_bins, bin_widths * 0.9)):
        left = x - width / 2
        right = x + width / 2
        
        # Draw gap regions
        if y < bin_centers[i]:  # Below the line (underconfident)
            gap_height = bin_centers[i] - y
            ax.add_patch(plt.Rectangle((left, y), width, gap_height, 
                                      color='red', alpha=0.3))
        elif y > bin_centers[i]:  # Above the line (overconfident)
            gap_height = y - bin_centers[i]
            ax.add_patch(plt.Rectangle((left, bin_centers[i]), width, gap_height, 
                                      color='red', alpha=0.3))
    
    # Add a patch to the legend
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color='red', alpha=0.3, label='Gap'))
    
    # Calculate and display the total error
    ece = compute_ece(y_prob, y_true, n_bins)
    total_error = sum(count_in_bins[i] * abs(acc_in_bins[i] - bin_centers[i]) 
                      for i in range(n_bins)) / sum(count_in_bins) * 100
    
    # Add an error box
    error_text = f"Error={total_error:.1f}"
    ax.text(0.7, 0.1, error_text, transform=ax.transAxes, 
            bbox=dict(facecolor='lightblue', alpha=0.8), fontsize=14)
    
    # Configure the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Calibration Curve')
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return ece

def calibrate_ensemble(ensemble_logits, y_true, device):
    """Apply temperature scaling to calibrate ensemble predictions."""
    # Convert data to tensors
    if isinstance(ensemble_logits, torch.Tensor):
        ensemble_logits_tensor = ensemble_logits.clone().detach().to(device)
    else:
        ensemble_logits_tensor = torch.tensor(ensemble_logits).to(device)
    
    if isinstance(y_true, torch.Tensor):
        y_true_tensor = y_true.clone().detach().to(device).long()
    else:
        y_true_tensor = torch.tensor(y_true, dtype=torch.long).to(device)
    
    # Initialize and fit temperature scaler
    temperature_scaler = TemperatureScaler()
    temperature_scaler.calibrate(ensemble_logits_tensor, y_true_tensor)
    
    # Apply calibration
    with torch.no_grad():
        calibrated_logits = temperature_scaler(ensemble_logits_tensor)
        calibrated_probs = torch.softmax(calibrated_logits, dim=1)
    
    # Return both original and calibrated probabilities
    return calibrated_probs.cpu().numpy(), temperature_scaler.temperature.item()

def analyze_ensemble(model, samples, X, y, device, history, save_dir=None, apply_calibration=True):
    """Analyze ensemble performance and generate plots.
    
    Args:
        model: The model template
        samples: List of model state dictionaries
        X: Input data
        y: True labels
        device: Device to run computations on
        history: Training history dictionary
        save_dir: Directory to save results (optional)
        apply_calibration: Whether to apply temperature scaling calibration
    """
    if not samples:
        return None
    
    X = torch.FloatTensor(X).to(device)
    y = torch.LongTensor(y).to(device)
    
    results = []
    ensemble_logits = []
    individual_probs = []
    
    # Calculate sampling epochs from history
    num_epochs = len(history['train_loss'])
    sampling_percentage = len(samples) / num_epochs
    burn_in_epochs = int(num_epochs * (1 - sampling_percentage))
    sampling_interval = (num_epochs - burn_in_epochs) // len(samples)
    sampling_epochs = [burn_in_epochs + i * sampling_interval for i in range(len(samples))]
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            model.load_state_dict(sample['state_dict'])
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            
            # Compute metrics for individual model
            metrics = {
                'model': f'sample_{i}',
                'type': 'individual',
                'accuracy': accuracy_score(y_np, predictions),
                'precision': precision_score(y_np, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_np, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y_np, predictions, average='weighted', zero_division=0),
                'ece': compute_ece(probs.cpu().numpy(), y_np)
            }
            results.append(metrics)
            ensemble_logits.append(outputs)
            individual_probs.append(probs.cpu().numpy())
    
    # Combine ensemble predictions
    ensemble_logits_combined = torch.stack(ensemble_logits).mean(dim=0)
    ensemble_probs = torch.softmax(ensemble_logits_combined, dim=1)
    y_np = y.cpu().numpy()
    
    # Apply calibration if requested
    calibrated_probs = None
    temperature = None
    if apply_calibration:
        calibrated_probs, temperature = calibrate_ensemble(ensemble_logits_combined, y_np, device)
        # Use calibrated probabilities for final predictions
        final_probs = calibrated_probs
    else:
        # Use uncalibrated ensemble probabilities
        final_probs = ensemble_probs.cpu().numpy()
    
    # Compute predictions from final probabilities
    predictions = np.argmax(final_probs, axis=1)
    
    # Add ensemble metrics to results
    ensemble_metrics = {
        'model': 'ensemble',
        'type': 'ensemble',
        'accuracy': accuracy_score(y_np, predictions),
        'precision': precision_score(y_np, predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_np, predictions, average='weighted', zero_division=0),
        'f1': f1_score(y_np, predictions, average='weighted', zero_division=0),
        'ece': compute_ece(final_probs, y_np)
    }
    
    if apply_calibration:
        ensemble_metrics['temperature'] = temperature
    
    results.append(ensemble_metrics)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to CSV
        df = pd.DataFrame(results)
        df.to_csv(save_dir / 'model_analysis.csv', index=False)
        
        # Generate and save plots
        plot_training_history(history, sampling_epochs, save_dir / 'training_history.png')
        
        # Save both calibrated and uncalibrated plots if calibration was applied
        if apply_calibration:
            plot_calibration_curve(y_np, ensemble_probs.cpu().numpy(), 
                                  save_dir / 'calibration_curve_uncalibrated.png')
            plot_calibration_curve(y_np, final_probs, 
                                  save_dir / 'calibration_curve.png')
            
            # Save calibration comparison plot
            plt.figure(figsize=(10, 6))
            confidences_uncal = np.max(ensemble_probs.cpu().numpy(), axis=1)
            confidences_cal = np.max(final_probs, axis=1)
            
            plt.hist(confidences_uncal, alpha=0.5, bins=20, label=f'Uncalibrated (T=1.0)')
            plt.hist(confidences_cal, alpha=0.5, bins=20, label=f'Calibrated (T={temperature:.2f})')
            
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Effect of Temperature Scaling on Confidence Distribution')
            plt.legend()
            plt.savefig(save_dir / 'confidence_comparison.png')
            plt.close()
        else:
            plot_calibration_curve(y_np, final_probs, save_dir / 'calibration_curve.png')
    
    return results

