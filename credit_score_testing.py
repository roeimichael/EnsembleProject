import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve


X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                           n_redundant=0, n_classes=3, random_state=42)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std


with pm.Model() as bayes_model:
    intercept = pm.Normal("intercept", mu=0, sigma=1, shape=3)
    coefs = pm.Normal("coefs", mu=0, sigma=1, shape=(5, 3))
    logits = pm.math.dot(X_train_scaled, coefs) + intercept
    p_unclipped = pm.math.softmax(logits)
    p = pm.Deterministic("p", pm.math.clip(p_unclipped, 1e-6, 1 - 1e-6))
    likelihood = pm.Categorical("likelihood", p=p, observed=y_train)
    print(likelihood)
    start_vals = pm.find_MAP()
    print("MAP starting values:", start_vals)
    trace = pm.sample(1000, tune=1000, target_accept=0.9, start=start_vals, cores=1)

# ----------------------------
ensemble_indices = np.random.choice(len(trace["intercept"]), size=10, replace=False)
ensemble_intercepts = trace["intercept"][ensemble_indices]  # shape: (10, 3)
ensemble_coefs = trace["coefs"][ensemble_indices]  # shape: (10, 5, 3)

def predict_ensemble(X, ensemble_intercepts, ensemble_coefs):
    preds = []
    for i in range(len(ensemble_intercepts)):
        logits = X.dot(ensemble_coefs[i]) + ensemble_intercepts[i]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds.append(prob)
    preds_avg = np.mean(preds, axis=0)
    return preds_avg


ensemble_probs = predict_ensemble(X_test_scaled, ensemble_intercepts, ensemble_coefs)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
print("Bayesian Ensemble Accuracy:", ensemble_accuracy)


clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
classical_probs = clf.predict_proba(X_test)
classical_preds = clf.predict(X_test)
classical_accuracy = accuracy_score(y_test, classical_preds)
print("Classical Logistic Regression Accuracy:", classical_accuracy)


def plot_calibration_curve(probs, y_true, model_name, n_bins=10):
    """
    Plots a calibration curve (reliability diagram) using the maximum predicted probability
    as the model’s confidence.
    """
    confidences = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == y_true).astype(int)
    fraction_of_positives, mean_predicted_value = calibration_curve(correct, confidences, n_bins=n_bins)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration")
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction of correct predictions")
    plt.title(f"Calibration Plot: {model_name}")
    plt.legend()
    plt.show()


plot_calibration_curve(ensemble_probs, y_test, "Bayesian Ensemble")
plot_calibration_curve(classical_probs, y_test, "Classical Logistic Regression")


def compute_ece(probs, y_true, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(bin_mask):
            bin_confidence = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(correct[bin_mask])
            ece += np.abs(bin_accuracy - bin_confidence) * np.sum(bin_mask) / len(confidences)
    return ece


ensemble_ece = compute_ece(ensemble_probs, y_test, n_bins=10)
classical_ece = compute_ece(classical_probs, y_test, n_bins=10)
print("Bayesian Ensemble ECE:", ensemble_ece)
print("Classical Logistic Regression ECE:", classical_ece)
