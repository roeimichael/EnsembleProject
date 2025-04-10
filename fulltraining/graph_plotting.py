import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from decorators import log_and_handle_errors, LogLevel
from Models import BayesianNN
import torch.nn as nn
from sklearn.calibration import calibration_curve
import torch.nn.functional as F

# Set up logging
logger = logging.getLogger(__name__)

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def ensure_results_dir() -> Path:
    """Create and return the directory for storing results (graphs and data)."""
    try:
        # Get the directory of the current file (fulltraining folder)
        current_dir = Path(__file__).parent
        # Create results directory inside fulltraining
        save_dir = current_dir / "results"
        save_dir.mkdir(exist_ok=True)
        return save_dir
    except Exception as e:
        logger.error(f"Error creating results directory: {e}")
        raise

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def plot_parameter_variance(models: List[nn.Module]) -> None:
    """Plot histogram of parameter variances across Bayesian samples."""
    all_params = []
    for model in models:
        flat_params = torch.cat([p.flatten() for p in model.parameters()])
        all_params.append(flat_params.detach().cpu().numpy())
    
    all_params = np.array(all_params)
    param_vars = np.var(all_params, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.hist(param_vars, bins=50, alpha=0.7)
    plt.title("Parameter Variance Distribution")
    plt.xlabel("Variance")
    plt.ylabel("Count")
    
    # Save the plot
    save_dir = ensure_results_dir()
    save_path = save_dir / "parameter_variance.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved parameter variance plot to {save_path}")
    plt.close()

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def plot_prediction_variance(models: List[nn.Module], X: torch.Tensor, device: torch.device) -> None:
    """Plot histogram of prediction standard deviations across Bayesian samples."""
    all_preds = []
    with torch.no_grad():
        X = X.to(device)
        for model in models:
            model.eval()
            pred = model(X).cpu().numpy().flatten()
            all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    pred_stds = np.std(all_preds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.hist(pred_stds, bins=50, alpha=0.7)
    plt.title("Prediction Standard Deviation Distribution")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Count")
    
    # Save the plot
    save_dir = ensure_results_dir()
    save_path = save_dir / "prediction_variance.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved prediction variance plot to {save_path}")
    plt.close()

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def plot_calibration_curve(
    model_name: str,
    probas: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> None:
    """Plot calibration curve comparing predicted probabilities with actual outcomes."""
    from helper_functions import compute_calibration_metrics
    result = compute_calibration_metrics(probas, labels, n_bins)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot calibration curve in the top subplot
    bar_width = 0.08
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Perfect Calibration')
    
    # Plot the actual outputs (red bars) for bins with predictions
    mask = result['bin_counts'] > 0
    bin_centers = np.linspace(0, 1, n_bins)
    ax1.bar(bin_centers[mask], result['bin_correct'][mask] / np.maximum(result['bin_counts'][mask], 1),
            width=bar_width, label='Confidence',
            color='red', alpha=0.8, edgecolor='black')
    
    # Plot the gap between ideal and actual (blue hatched bars)
    for i, (center, correct, count) in enumerate(zip(bin_centers, result['bin_correct'], result['bin_counts'])):
        if count > 0:
            actual_prob = correct / count
            if actual_prob < center:
                # If actual is less than expected, plot gap above
                ax1.bar(center, center - actual_prob,
                       bottom=actual_prob, width=bar_width,
                       color='blue', alpha=0.3, hatch='/', edgecolor='blue',
                       label='Gap' if i == 0 else None)
            else:
                # If actual is more than expected, plot gap below
                ax1.bar(center, actual_prob - center,
                       bottom=center, width=bar_width,
                       color='blue', alpha=0.3, hatch='/', edgecolor='blue',
                       label='Gap' if i == 0 else None)
    
    # Add error metrics
    ax1.text(0.05, 0.95,
             f'ECE: {result["ece"]*100:.1f}%\n'
             f'Mean Confidence: {result["conf_mean"]:.3f}\n'
             f'Accuracy: {np.sum(result["bin_correct"])/np.sum(result["bin_counts"]):.3f}',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             verticalalignment='top')
    
    # Customize the calibration plot
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Actual Probability")
    ax1.set_title(f"Calibration Curve - {model_name}")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    
    # Plot prediction distribution in the bottom subplot
    hist_counts, hist_bins = np.histogram(probas, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

    # Plot histogram of prediction probabilities
    ax2.bar(bin_centers, hist_counts, width=(1/n_bins)*0.9,
            color='lightblue', edgecolor='black')

    # Add count labels
    for i, count in enumerate(hist_counts):
        if count > 0:
            ax2.text(bin_centers[i], count, f'{int(count)}',
                    ha='center', va='bottom')
    
    # Add distribution statistics
    stats_text = (
        f'Total predictions: {len(probas)}\n'
        f'Mean: {np.mean(probas):.3f}\n'
        f'Std: {np.std(probas):.3f}'
    )
    ax2.text(0.02, 0.95, stats_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Number of Predictions")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(-0.02, 1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_dir = ensure_results_dir()
    filename = f'calibration_curve_{model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")}.png'
    save_path = save_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved calibration curve to {save_path}")
    plt.close()

@log_and_handle_errors(log_level=LogLevel.DEBUG)
def plot_model_analysis(
    models: List[nn.Module], 
    X: torch.Tensor, 
    y: torch.Tensor,
    lr: float, 
    noise_std: float, 
    device: torch.device
) -> None:
    """Create comprehensive analysis plots for Bayesian models."""
    try:
        # Ensure consistent dimensions
        n_samples = min(len(X), len(y))
        X = X[:n_samples]
        y = y[:n_samples]
        
        logger.info(f"Plotting model analysis with {n_samples} samples")
        
        # Plot parameter variances
        plot_parameter_variance(models)
        
        # Plot prediction variances
        plot_prediction_variance(models, X, device)
        
        # Plot calibration curve for ensemble
        all_preds = []
        with torch.no_grad():
            X = X.to(device)
            for model in models:
                model.eval()
                outputs = model(X)
                probs = F.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())
        
        # Average predictions
        ensemble_pred = np.mean(all_preds, axis=0)
        y_true = y.cpu().numpy()
        
        # Plot calibration curve
        plot_calibration_curve("BayesianNN_Ensemble", ensemble_pred, y_true)
        
    except Exception as e:
        logger.error(f"Error in plot_model_analysis: {str(e)}")
        raise 