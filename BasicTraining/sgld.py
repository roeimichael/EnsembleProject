import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import pandas as pd

class LinearModel(nn.Module):
    """Simple linear model with Gaussian prior initialization."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.linear.bias, mean=0.0, std=1.0)
    
    def forward(self, x):
        return self.linear(x)

class SGLDTrainer:
    """Stochastic Gradient Langevin Dynamics trainer with ensemble sampling."""
    def __init__(self, input_dim, num_classes, prior_std=1.0, temperature=1.0, device="cuda"):
        self.device = torch.device(device)
        self.model = LinearModel(input_dim, num_classes).to(self.device)
        self.prior_std = prior_std
        self.temperature = temperature
        self.samples = []
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [],
            'train_entropy': [], 'val_entropy': []
        }
    
    def train(self, X_train, y_train, X_val, y_val, num_epochs=2000, batch_size=32, learning_rate=0.001):
        """Train model using SGLD with burn-in and sampling phases."""
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            predictions = []
            total_entropy = 0
            
            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Add prior loss with proper scaling
                prior_loss = 0
                for param in self.model.parameters():
                    prior_loss += torch.sum(param ** 2) / (2 * self.prior_std ** 2)
                loss += prior_loss
                
                loss.backward()
                
                # SGLD update with proper noise scaling
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise_std = np.sqrt(2 * learning_rate * self.temperature / len(train_loader))
                        noise = torch.randn_like(param) * noise_std
                        param.data -= learning_rate * param.grad + noise
                
                self.model.zero_grad()
                
                # Calculate metrics
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                total_entropy += entropy.item()
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            # Calculate epoch metrics
            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(y_train.cpu().numpy(), predictions)
            train_entropy = total_entropy / len(train_loader)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(X_val, y_val)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_entropy'].append(train_entropy)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_entropy'].append(val_metrics['entropy'])
            
            # Collect sample after burn-in
            if epoch >= 1000 and (epoch - 1000) % 100 == 0:
                self._collect_sample()
                logging.info(f"Collected sample at epoch {epoch}")
            
            # Log progress
            if (epoch + 1) % 100 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Train Entropy: {train_entropy:.4f}, Val Entropy: {val_metrics['entropy']:.4f}"
                )
    
    def evaluate(self, X, y):
        """Evaluate model performance on given data."""
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            loss = nn.CrossEntropyLoss()(outputs, y)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            y = y.cpu().numpy()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0),
            'entropy': entropy
        }
    
    def _collect_sample(self):
        """Collect a sample of model parameters."""
        sample = {
            'state_dict': {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        }
        self.samples.append(sample)
    
    def analyze_models(self, X, y, save_path=None):
        """Analyze individual models and ensemble performance."""
        if not self.samples:
            logging.warning("No samples collected, skipping analysis")
            return None
        
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        results = []
        
        # Evaluate individual models
        for i, sample in enumerate(self.samples):
            self.model.load_state_dict(sample['state_dict'])
            metrics = self.evaluate(X, y)
            results.append({
                'model': f'sample_{i}',
                'type': 'individual',
                **metrics
            })
        
        # Evaluate ensemble
        ensemble_probs = []
        with torch.no_grad():
            for sample in self.samples:
                self.model.load_state_dict(sample['state_dict'])
                outputs = self.model(X)
                probs = torch.softmax(outputs, dim=1)
                ensemble_probs.append(probs)
        
        ensemble_probs = torch.stack(ensemble_probs).mean(dim=0)
        predictions = torch.argmax(ensemble_probs, dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        
        ensemble_entropy = -torch.sum(ensemble_probs * torch.log(ensemble_probs + 1e-10), dim=1).mean().item()
        
        results.append({
            'model': 'ensemble',
            'type': 'ensemble',
            'loss': float('nan'),  # Not applicable for ensemble
            'accuracy': accuracy_score(y_np, predictions),
            'precision': precision_score(y_np, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_np, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_np, predictions, average='weighted', zero_division=0),
            'entropy': ensemble_entropy
        })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
        
        return df

def plot_training_comparison(history, save_path=None):
    """Plot training history including loss, accuracy, and entropy."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy over time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_entropy'], label='Train')
    plt.plot(history['val_entropy'], label='Validation')
    plt.title('Entropy over time')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.legend()
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_predictions_comparison(trainer, X, y, save_path=None):
    """Plot prediction distribution and uncertainty visualization."""
    if not trainer.samples:
        logging.warning("No samples collected, skipping prediction plot")
        return
    
    # Get predictions from all samples
    all_predictions = []
    all_probs = []
    
    X = torch.FloatTensor(X).to(trainer.device)
    with torch.no_grad():
        for sample in trainer.samples:
            trainer.model.load_state_dict(sample['state_dict'])
            outputs = trainer.model(X)
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(torch.argmax(probs, dim=1))
            all_probs.append(probs)
    
    # Convert to numpy for plotting
    all_predictions = torch.stack(all_predictions).cpu().numpy()
    all_probs = torch.stack(all_probs).cpu().numpy()
    ensemble_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(ensemble_probs, axis=1)
    entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot true class distribution
    plt.subplot(1, 3, 1)
    for i in range(len(np.unique(y))):
        mask = y == i
        plt.scatter(X[mask, 0].cpu(), X[mask, 1].cpu(), alpha=0.5, label=f'Class {i}')
    plt.title('True Class Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot predicted class distribution
    plt.subplot(1, 3, 2)
    for i in range(len(np.unique(predictions))):
        mask = predictions == i
        plt.scatter(X[mask, 0].cpu(), X[mask, 1].cpu(), alpha=0.5, label=f'Predicted {i}')
    plt.title('Ensemble Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot uncertainty
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=entropy, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Entropy')
    plt.title('Prediction Uncertainty')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close() 