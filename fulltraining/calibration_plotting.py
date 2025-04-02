"""
Calibration Plotting Module

This module provides functions to assess and visualize model calibration through various metrics:
- Expected Calibration Error (ECE)
- Adaptive Expected Calibration Error (AECE)
- Conditional Correlation
- Brier Score
- Gap-style calibration curves
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Tuple, List


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute standard Expected Calibration Error (ECE) using fixed-width binning.
    ECE measures the difference between predicted probabilities and actual accuracies.
    
    Args:
        probs: Model predicted probabilities of shape (n_samples, n_classes)
        labels: True class labels of shape (n_samples,)
        num_bins: Number of bins for discretizing predictions
        
    Returns:
        float: Expected Calibration Error score
    """
    n = len(labels)
    confidences = np.max(probs, axis=1)  # Get confidence scores
    predictions = np.argmax(probs, axis=1)  # Get predicted classes
    correct = (predictions == labels)  # Binary array of correct predictions

    # Create bins and initialize ECE
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    # Compute ECE across bins
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Handle the edge case for the last bin
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper) if i == num_bins - 1 else (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)

    return ece


def compute_adaptive_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute adaptive Expected Calibration Error using equal-mass binning.
    This version ensures each bin contains roughly the same number of samples.
    
    Args:
        probs: Model predicted probabilities of shape (n_samples, n_classes)
        labels: True class labels of shape (n_samples,)
        num_bins: Number of bins for discretizing predictions
        
    Returns:
        float: Adaptive Expected Calibration Error score
    """
    n = len(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    # Sort samples by confidence
    sorted_indices = np.argsort(confidences)
    confidences = confidences[sorted_indices]
    correct = correct[sorted_indices]

    # Create equal-mass bins
    samples_per_bin = n // num_bins
    remainder = n % num_bins
    bin_sizes = np.full(num_bins, samples_per_bin)
    bin_sizes[:remainder] += 1  # Distribute remaining samples
    bin_boundaries = np.cumsum(bin_sizes)

    # Compute adaptive ECE
    ece = 0.0
    start = 0
    for end in bin_boundaries:
        bin_confidence = np.mean(confidences[start:end])
        bin_accuracy = np.mean(correct[start:end])
        bin_size = end - start
        ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)
        start = end

    return ece


def compute_conditional_correlation(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Pearson correlation between model confidence and prediction correctness.
    A higher correlation indicates better calibration.
    
    Args:
        probs: Model predicted probabilities of shape (n_samples, n_classes)
        labels: True class labels of shape (n_samples,)
        
    Returns:
        float: Pearson correlation coefficient
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(float)
    return pearsonr(confidences, correctness)[0]


def plot_gap_style_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
    num_bins: int = 10
) -> Tuple[float, List[float], List[float]]:
    """
    Plot calibration curve showing the gap between predicted confidence and actual accuracy.
    Also displays confidence distribution and calibration error.
    
    Args:
        probs: Model predicted probabilities of shape (n_samples, n_classes)
        labels: True class labels of shape (n_samples,)
        model_name: Name of the model for plot title and filename
        num_bins: Number of bins for discretizing predictions
        
    Returns:
        Tuple containing:
        - float: Expected Calibration Error
        - List[float]: Bin accuracies
        - List[float]: Bin confidences
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    # Print debug information about confidence distribution
    print(f"\nConfidence Distribution Summary for {model_name}:")
    print(f"Range: {np.min(confidences):.3f} to {np.max(confidences):.3f}")
    print(f"Mean: {np.mean(confidences):.3f}")
    print(f"Std: {np.std(confidences):.3f}")
    
    # Create histogram for confidence distribution
    hist, edges = np.histogram(confidences, bins=np.linspace(0.0, 1.0, num_bins + 1))
    print("\nConfidence Distribution by Bin:")
    for i in range(num_bins):
        print(f"Bin {i+1:2d} ({edges[i]:.2f}–{edges[i+1]:.2f}): {hist[i]:4d} samples")

    # Compute calibration metrics
    n = len(labels)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    ece = 0.0

    # Compute metrics for each bin
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper) if i == num_bins - 1 else (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
        else:
            bin_acc = bin_conf = bin_centers[i]  # Use bin center for empty bins
            
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_sizes.append(bin_size)

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Plot calibration curve (top subplot)
    bar_width = 1.0 / (num_bins + 1)
    for i in range(num_bins):
        # Plot confidence bars
        ax1.bar(bin_centers[i], bin_confidences[i], 
                width=bar_width, color='blue', edgecolor='black',
                label='Confidence' if i == 0 else None)
        
        # Plot accuracy-confidence gaps
        gap = np.abs(bin_confidences[i] - bin_accuracies[i])
        if gap > 0:
            bottom = min(bin_confidences[i], bin_accuracies[i])
            ax1.bar(bin_centers[i], gap, width=bar_width,
                   bottom=bottom, color='red', alpha=0.3,
                   edgecolor='darkred', hatch='/',
                   label='Calibration Gap' if i == 0 else None)

    # Add perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Empirical Probability')
    ax1.set_title(f'Calibration Curve - {model_name}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add ECE score and other metrics
    metrics_text = (
        f'ECE = {ece:.3f}\n'
        f'Mean Confidence = {np.mean(confidences):.3f}\n'
        f'Mean Accuracy = {np.mean(correct):.3f}'
    )
    ax1.text(0.05, 0.95, metrics_text,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             verticalalignment='top')

    # Plot confidence distribution (bottom subplot)
    ax2.bar(bin_centers, bin_sizes, width=bar_width,
            color='lightblue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)

    # Add sample counts on bars
    for i, count in enumerate(bin_sizes):
        if count > 0:
            ax2.text(bin_centers[i], count, str(count),
                    ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the plot to the results directory
    from pathlib import Path
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    filename = f'calibration_curve_{clean_model_name}.png'
    save_path = results_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return ece, bin_accuracies, bin_confidences


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Brier score (mean squared error of predicted probabilities).
    Lower scores indicate better calibrated predictions.
    
    Args:
        probs: Model predicted probabilities of shape (n_samples, n_classes)
        labels: True class labels of shape (n_samples,)
        
    Returns:
        float: Brier score
    """
    n_classes = probs.shape[1]
    one_hot_labels = np.eye(n_classes)[labels]
    return np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    num_samples = 1000
    num_classes = 20  # Increased number of classes to get lower confidence values

    # Generate synthetic probabilities with varying confidence levels
    raw_logits = np.random.randn(num_samples, num_classes)
    scaled_logits = raw_logits * np.random.uniform(0.5, 5.0, size=(num_samples, 1))
    probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)

    # Add some extreme cases for testing
    extreme_cases = np.zeros((4, num_classes))
    # Very low confidence - almost uniform distribution
    extreme_cases[0] = np.array([1/num_classes + 0.001] * num_classes)
    extreme_cases[0, 0] = 1/num_classes - 0.001 * (num_classes-1)  # Adjust to sum to 1
    
    # Low confidence
    extreme_cases[1, 0] = 0.1  # Main class
    extreme_cases[1, 1:] = 0.9/(num_classes-1)  # Rest distributed evenly
    
    # High confidence
    extreme_cases[2, 0] = 0.95  # Main class
    extreme_cases[2, 1:] = 0.05/(num_classes-1)  # Rest distributed evenly
    
    # Medium confidence
    extreme_cases[3, 0] = 0.4  # Main class
    extreme_cases[3, 1:] = 0.6/(num_classes-1)  # Rest distributed evenly
    
    # Replace first few samples with extreme cases
    num_extreme = len(extreme_cases)
    probs[:num_extreme] = extreme_cases
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_samples)

    # Compute and print all metrics
    print("\nCalibration Metrics:")
    print(f"Standard ECE: {compute_ece(probs, labels, num_bins=15):.3f}")
    print(f"Adaptive ECE: {compute_adaptive_ece(probs, labels, num_bins=15):.3f}")
    print(f"Conditional Correlation: {compute_conditional_correlation(probs, labels):.3f}")
    print(f"Brier Score: {brier_score(probs, labels):.3f}")

    # Plot calibration curve
    ece, accuracies, confidences = plot_gap_style_calibration_curve(probs, labels, model_name="Synthetic Data", num_bins=15)
    plt.show()