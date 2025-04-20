import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, List, Optional
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
import seaborn as sns
import os
from sklearn.datasets import load_wine

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_path: str = 'training.log'):
    """Configure logging to file and console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def load_dataset():
    """Load and preprocess the wine dataset."""
    data = load_wine()
    X = data.data
    y = data.target
    return X, y

def prepare_data(X, y, train_size=0.8, random_state=42):
    """Split and scale the dataset."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model and return the best model and training history."""
    try:
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0
        best_model = None
        patience = config['training'].get('patience', 10)
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and config['training'].get('early_stopping', True):
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{config['training']['epochs']}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        return model, history
    except Exception as e:
        raise RuntimeError(f"Error during training: {str(e)}")

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate model performance."""
    try:
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted'),
            'recall': recall_score(all_targets, all_preds, average='weighted'),
            'f1': f1_score(all_targets, all_preds, average='weighted')
        }
        
        return metrics, all_probs
    except Exception as e:
        raise RuntimeError(f"Error during evaluation: {str(e)}")

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training metrics over time."""
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Error plotting training history: {str(e)}")

def plot_calibration_curve(y_true, y_prob, save_path, n_bins=10):
    """Plot model calibration curve."""
    plt.figure(figsize=(8, 6))
    
    for i in range(y_prob.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true == i, y_prob[:, i], n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_model(model: nn.Module, path: str):
    """Save model to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
    except Exception as e:
        raise RuntimeError(f"Error saving model: {str(e)}")

def load_model(model: nn.Module, path: str):
    """Load model from disk."""
    try:
        model.load_state_dict(torch.load(path))
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}") 