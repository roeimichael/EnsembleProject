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
    
    # Plot the actual outputs (blue bars) only for bins with predictions
    mask = result.bin_counts > 0
    ax1.bar(result.bin_centers[mask], result.prob_true[mask],
            width=bar_width, label='Outputs',
            color='blue', edgecolor='black')
    
    # Plot the gap between ideal and actual (hollow red bars)
    for i, (center, true, count) in enumerate(zip(result.bin_centers, result.prob_true, result.bin_counts)):
        if count > 0:  # Only plot gaps for bins with predictions
            if true < center:
                # If actual is less than expected, plot gap above
                ax1.bar(center, center - true,
                       bottom=true, width=bar_width,
                       color='none', edgecolor='red', label='Gap' if i == 0 else None,
                       linestyle='-', linewidth=1)
            else:
                # If actual is more than expected, plot gap below
                ax1.bar(center, true - center,
                       bottom=center, width=bar_width,
                       color='none', edgecolor='red', label='Gap' if i == 0 else None,
                       linestyle='-', linewidth=1)
    
    # Plot the perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Perfect Calibration')
    
    # Add error text in bottom right
    ax1.text(0.75, 0.1, f'ECE={result.ece*100:.1f}%',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             transform=ax1.transAxes)
    
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
    ax2.bar(bin_centers, hist_counts, width=(1/n_bins)*0.9,  # width is smaller than bin width for spacing
            color='lightblue', edgecolor='black')

    # Add count labels
    for i, count in enumerate(hist_counts):
        if count > 0:
            ax2.text(bin_centers[i], count, f'{int(count)}',
                    ha='center', va='bottom')
    
    # Add total predictions text and distribution statistics
    total_predictions = len(probas)
    stats_text = (
        f'Total predictions: {total_predictions}\n'
        f'Mean: {np.mean(probas):.3f}\n'
        f'Std: {np.std(probas):.3f}\n'
        f'Min: {np.min(probas):.3f}\n'
        f'Max: {np.max(probas):.3f}'
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
            pred = model(X).cpu().numpy()
            all_preds.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    y_true = y.numpy().flatten()
    
    # Ensure predictions are 2D
    if len(ensemble_pred.shape) == 1:
        ensemble_pred = ensemble_pred.reshape(-1, 1)
    
    # Plot calibration curve
    plot_calibration_curve("Ensemble", ensemble_pred, y_true) 