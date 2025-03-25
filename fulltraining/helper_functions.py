import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.calibration import calibration_curve
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
def train_simplenn(model, dataloader, epochs=50, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def sgld_training(model, dataloader, epochs=50, lr=1e-3, noise_std=1e-4, burnin_ratio=0.5):
    model.train()
    samples = []
    criterion = nn.BCELoss()
    params = list(model.parameters())
    burnin_epoch = int(epochs * burnin_ratio)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            model.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            for p in params:
                grad = p.grad
                noise = torch.randn_like(p) * noise_std
                p.data += -0.5 * lr * grad + noise

        if epoch >= burnin_epoch:
            sampled_weights = [(p.data.clone()) for p in model.parameters()]
            samples.append(sampled_weights)

    return samples

def compute_ece(probas, labels, n_bins=10):
    prob_true, prob_pred = calibration_curve(labels, probas, n_bins=n_bins, strategy='uniform')

    # Compute bin assignments and counts
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probas, bin_edges[1:-1], right=True)
    bin_counts = np.array([np.sum(bin_ids == i) for i in range(n_bins)])

    # Use only available bins
    used_bins = min(len(prob_true), len(prob_pred), len(bin_counts))
    bin_proportions = bin_counts[:used_bins] / len(probas)
    ece = np.sum(np.abs(prob_pred[:used_bins] - prob_true[:used_bins]) * bin_proportions)
    return ece
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = (outputs >= 0.5).float()
        probas = outputs.numpy().flatten()
        labels = y_test.numpy().flatten()
        preds_np = preds.numpy().flatten()

        acc = accuracy_score(labels, preds_np)
        prec = precision_score(labels, preds_np)
        f1 = f1_score(labels, preds_np)

        # Fix ECE: match histogram bins and number of calibration bins
        n_bins = 10
        prob_true, prob_pred = calibration_curve(labels, probas, n_bins=n_bins, strategy='uniform')

        # Recalculate bin proportions to match number of bins in calibration_curve
        bin_counts, _ = np.histogram(probas, bins=n_bins, range=(0, 1))
        bin_proportions = bin_counts[:len(prob_true)] / len(probas)

        ece = compute_ece(probas, labels)

        return {"Accuracy": acc, "Precision": prec, "F1 Score": f1, "ECE": ece}

def save_results(results, filename):
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = "Model"
    results_df.to_csv(filename)
    print(f"Results saved to {filename}")

def plot_parameter_variance(bayes_models):
    all_params = []
    for model in bayes_models:
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        all_params.append(flat_params.detach().numpy())
    all_params = np.stack(all_params)
    variances = np.var(all_params, axis=0)

    plt.figure(figsize=(10, 4))
    plt.hist(variances, bins=50, alpha=0.75)
    plt.title("Histogram of Parameter Variances Across Bayesian Samples")
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_prediction_variance(bayes_models, X_test_tensor):
    predictions = []
    with torch.no_grad():
        for model in bayes_models:
            preds = model(X_test_tensor)
            predictions.append(preds.numpy().flatten())
    predictions = np.stack(predictions)
    prediction_std = np.std(predictions, axis=0)

    plt.figure(figsize=(10, 4))
    plt.hist(prediction_std, bins=40, alpha=0.75)
    plt.title("Prediction Standard Deviation Across Bayesian Samples (Uncertainty Histogram)")
    plt.xlabel("Std Dev of Predictions")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_calibration_curve(model_name, probas, labels, n_bins=10):
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(labels, probas, n_bins=n_bins, strategy='uniform')

    # Compute bin centers
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Compute ECE manually for plot display
    bin_ids = np.digitize(probas, bin_edges[1:-1], right=True)
    bin_counts = np.array([np.sum(bin_ids == i) for i in range(n_bins)])
    used_bins = min(len(prob_true), len(prob_pred), len(bin_counts))
    bin_proportions = bin_counts[:used_bins] / len(probas)
    ece = np.sum(np.abs(prob_pred[:used_bins] - prob_true[:used_bins]) * bin_proportions)

    # Plot
    bar_width = 0.04
    plt.figure(figsize=(7, 5))

    plt.bar(bin_centers[:used_bins], prob_pred[:used_bins], width=bar_width, alpha=0.6, label='Expected', color='pink', edgecolor='black')
    plt.bar(bin_centers[:used_bins], prob_true[:used_bins], width=bar_width, alpha=0.8, label='Actual', color='blue', edgecolor='black')

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (y=x)', linewidth=1.5)

    # Labels and legend
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(title=f"Legend\nECE = {ece:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
