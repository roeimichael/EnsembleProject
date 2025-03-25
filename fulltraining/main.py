import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Models import SimpleNN, BayesianNN
from helper_functions import train_simplenn, sgld_training, evaluate_model, save_results, compute_ece,plot_calibration_curve
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score

# -------------------------
# Set configuration
lr = 0.001
noise_std = 0.001
epochs = 300
burnin_ratio = 0.5
# -------------------------

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train SimpleNN
simplenn = SimpleNN(X_train.shape[1])
simplenn = train_simplenn(simplenn, train_loader)

# Train BayesianNN with chosen SGLD config
print(f"\n🔍 Training BayesianNN with SGLD: lr={lr}, noise_std={noise_std}, epochs={epochs}, burnin_ratio={burnin_ratio}")
bayes_model = BayesianNN(X_train.shape[1])
posterior_samples = sgld_training(bayes_model, train_loader, epochs=epochs, lr=lr, noise_std=noise_std, burnin_ratio=burnin_ratio)

# Create list of Bayesian models from samples
bayes_models = []
for sample in posterior_samples[:10]:
    m = BayesianNN(X_train.shape[1])
    for p, sample_param in zip(m.parameters(), sample):
        p.data = sample_param.clone()
    bayes_models.append(m)

# Evaluate all models
model_results = {}
model_results["SimpleNN"] = evaluate_model(simplenn, X_test_tensor, y_test_tensor)

with torch.no_grad():
    simplenn_probs = simplenn(X_test_tensor).numpy().flatten()
    plot_calibration_curve("SimpleNN", simplenn_probs, y_test_tensor.numpy().flatten())

for idx, model in enumerate(bayes_models, start=1):
    model_results[f"BayesianNN_Sample_{idx}"] = evaluate_model(model, X_test_tensor, y_test_tensor)

# Evaluate ensemble
with torch.no_grad():
    probs_ensemble = torch.zeros_like(y_test_tensor)
    for model in bayes_models:
        probs_ensemble += model(X_test_tensor)
    probs_ensemble /= len(bayes_models)

    preds_ensemble = (probs_ensemble >= 0.5).float()
    probas = probs_ensemble.numpy().flatten()
    labels = y_test_tensor.numpy().flatten()
    preds_np = preds_ensemble.numpy().flatten()

    acc = accuracy_score(labels, preds_np)
    prec = precision_score(labels, preds_np)
    f1 = f1_score(labels, preds_np)
    ece = compute_ece(probas, labels)

    model_results["BayesianNN_Ensemble"] = {
        "Accuracy": acc,
        "Precision": prec,
        "F1 Score": f1,
        "ECE": ece
    }
plot_calibration_curve("BayesianNN Ensemble", probas, labels)

# Save results
filename = f'final_results_lr{lr}_noise{noise_std}_epochs{epochs}_burnin{int(burnin_ratio*100)}.csv'
save_results(model_results, filename)

# Plot parameter variance
all_params = []
for model in bayes_models:
    flat_params = torch.cat([p.view(-1) for p in model.parameters()])
    all_params.append(flat_params.detach().numpy())
all_params = np.stack(all_params)
variances = np.var(all_params, axis=0)

plt.figure(figsize=(10, 4))
plt.hist(variances, bins=50, alpha=0.75)
plt.title(f"Parameter Variance (lr={lr}, noise={noise_std}, epochs={epochs}, burnin={burnin_ratio})")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot prediction variance
predictions = []
with torch.no_grad():
    for model in bayes_models:
        preds = model(X_test_tensor)
        predictions.append(preds.numpy().flatten())
predictions = np.stack(predictions)
prediction_std = np.std(predictions, axis=0)

plt.figure(figsize=(10, 4))
plt.hist(prediction_std, bins=40, alpha=0.75)
plt.title(f"Prediction StdDev (lr={lr}, noise={noise_std}, epochs={epochs}, burnin={burnin_ratio})")
plt.xlabel("Std Dev of Predictions")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
