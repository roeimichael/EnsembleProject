import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from glob import glob
import warnings
from HelperFunctions import analyze_predictions

warnings.filterwarnings('ignore')


def load_datasets_and_shifted_targets(data_folder='data/', target_file='targets.csv'):
    targets_df = pd.read_csv(target_file)
    datasets = {}
    files = glob(os.path.join(data_folder, '*.csv'))
    for index, file_path in enumerate(sorted(files)):
        filename = os.path.basename(file_path).replace('.csv', '')
        if index < len(targets_df.columns):
            target_column = targets_df.columns[index]
            features = pd.read_csv(file_path).iloc[:, 2:].values
            target = targets_df[target_column].dropna().values

            datasets[filename] = (features, target)
            print(f"Dataset '{filename}' loaded with targets from column '{target_column}'.")
        else:
            print(f"No target available for '{filename}'.")

    return datasets

# Build and return SVM model
def build_svm_model(C_value=100, kernel='linear'):
    return SVC(C=C_value, kernel=kernel, probability=True)


# Calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


# Save predictions to CSV
def save_predictions(folder_path, y_test, predictions, predicted_classes):
    os.makedirs(folder_path, exist_ok=True)
    predictions_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Probability 0': predictions[:, 0],
        'Predicted Probability 1': predictions[:, 1],
        'Predicted Label': predicted_classes
    })
    predictions_output_path = os.path.join(folder_path, 'predictions.csv')
    predictions_df.to_csv(predictions_output_path, index=False)


# Plot probability analysis
def plot_probability_analysis(avg_tp, avg_fp, avg_tn, avg_fn, folder_path):
    labels = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    values = [avg_tp, avg_fp, avg_tn, avg_fn]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['green', 'red', 'blue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Average Prediction Probabilities')
    plt.ylabel('Average Probability')
    plt.savefig(os.path.join(folder_path, 'probability_analysis.png'))
    plt.close()


# Plot performance metrics
def plot_performance_metrics(accuracy, precision, recall, f1, folder_path):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['purple', 'cyan', 'magenta', 'yellow'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.ylabel('Metric Value')
    plt.savefig(os.path.join(folder_path, 'performance_metrics.png'))
    plt.close()


# Plot confidence curve
def plot_confidence_curve(y_true, predictions, folder_path, bins=10):
    predicted_probs = predictions[:, 1]
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    correct_counts = np.zeros(bins)
    total_counts = np.zeros(bins)
    for true_label, prob in zip(y_true, predicted_probs):
        bin_idx = np.digitize(prob, bin_edges) - 1
        total_counts[bin_idx] += 1
        if true_label == (prob >= 0.5):
            correct_counts[bin_idx] += 1
    accuracy_in_bin = correct_counts / total_counts
    accuracy_in_bin = np.nan_to_num(accuracy_in_bin)
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, accuracy_in_bin, marker='o', linestyle='-', color='blue')
    plt.title('Confidence Curve')
    plt.xlabel('Confidence (Predicted Probability)')
    plt.ylabel('Proportion of Correct Predictions')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, 'confidence_curve.png'))
    plt.close()


# Load datasets and targets
data_folder = 'data/'
target_file = 'targets.csv'
datasets = load_datasets_and_shifted_targets(data_folder=data_folder, target_file=target_file)


# Analyze each dataset and save results
for dataset_name, (X, y) in datasets.items():
    # Create results folder for each dataset
    results_folder = os.path.join('results', dataset_name)
    os.makedirs(results_folder, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train the SVM model
    model = build_svm_model()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict_proba(X_test)
    predicted_classes = model.predict(X_test)

    # Save predictions
    save_predictions(results_folder, y_test, predictions, predicted_classes)

    # Calculate metrics and plot
    avg_tp, avg_fp, avg_tn, avg_fn = analyze_predictions(y_test, predictions)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, predicted_classes)

    # Plot results
    plot_probability_analysis(avg_tp, avg_fp, avg_tn, avg_fn, results_folder)
    plot_performance_metrics(accuracy, precision, recall, f1, results_folder)
    plot_confidence_curve(y_test, predictions, results_folder, bins=5)

    print(f"Results for {dataset_name} saved in {results_folder}")