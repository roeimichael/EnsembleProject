import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


def load_and_process_data(path):
    data = pd.read_csv(path)
    features = data.iloc[:, 2:-1].values
    target = data.iloc[:, -1].values
    return np.array(features, dtype=np.float32), np.array(target, dtype=np.int32)


def build_svm_model(C_value, kernel='linear', gamma='scale'):
    model = SVC(C=C_value, kernel=kernel, probability=True, gamma=gamma)
    return model


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


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
    return predictions_output_path


def analyze_predictions(y_true, predictions):
    predicted_classes = np.argmax(predictions, axis=1)
    tp_probs, fp_probs, tn_probs, fn_probs = [], [], [], []
    for true_label, predicted_label, prob in zip(y_true, predicted_classes, predictions):
        if true_label == 1 and predicted_label == 1:
            tp_probs.append(prob[1])  # True positive, prob for 1
        elif true_label == 0 and predicted_label == 1:
            fp_probs.append(prob[1])  # False positive, prob for 1
        elif true_label == 0 and predicted_label == 0:
            tn_probs.append(prob[0])  # True negative, prob for 0
        elif true_label == 1 and predicted_label == 0:
            fn_probs.append(prob[0])  # False negative, prob for 0
    avg_tp = np.mean(tp_probs) if tp_probs else 0
    avg_fp = np.mean(fp_probs) if fp_probs else 0
    avg_tn = np.mean(tn_probs) if tn_probs else 0
    avg_fn = np.mean(fn_probs) if fn_probs else 0
    return avg_tp, avg_fp, avg_tn, avg_fn


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
