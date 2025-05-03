import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

class LinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.linear.bias, mean=0.0, std=1.0)
    
    def forward(self, x):
        return self.linear(x)

class SGLDTrainer:
    def __init__(self, input_dim, num_classes, prior_std=1.0, temperature=1.0, 
                 temp_min=0.1, temp_decay=1.0, device="cuda"):
        self.device = torch.device(device)
        self.model = LinearModel(input_dim, num_classes).to(self.device)
        self.prior_std = prior_std
        self.initial_temperature = temperature
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.samples = []
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [],
            'train_entropy': [], 'val_entropy': [],
            'temperature': []
        }
    
    def train(self, X_train, y_train, X_val, y_val, num_epochs=2000, batch_size=32, learning_rate=0.001, 
              sampling_percentage=0.5, num_ensemble_models=10, burn_in_percentage=None):
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        
        # Allow custom burn-in percentage
        if burn_in_percentage is None:
            burn_in_percentage = 1 - sampling_percentage
        
        burn_in_epochs = int(num_epochs * burn_in_percentage)
        sampling_epochs = num_epochs - burn_in_epochs
        sampling_interval = max(1, sampling_epochs // num_ensemble_models)
        
        print(f"Training for {num_epochs} epochs with:")
        print(f"  - Initial temperature: {self.temperature:.4f}")
        print(f"  - Temperature decay: {self.temp_decay:.4f}")
        print(f"  - Minimum temperature: {self.temp_min:.4f}")
        print(f"  - Prior std: {self.prior_std:.4f}")
        print(f"  - Burn-in period: {burn_in_epochs} epochs ({burn_in_percentage:.2f})")
        print(f"  - Sampling {num_ensemble_models} models every {sampling_interval} epochs after burn-in")
        
        for epoch in range(num_epochs):
            # Update temperature with decay
            if self.temp_decay < 1.0:
                self.temperature = max(
                    self.temp_min, 
                    self.initial_temperature * (self.temp_decay ** epoch)
                )
            
            self.model.train()
            total_loss, predictions, total_entropy = 0, [], 0
            
            for inputs, targets in train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                prior_loss = sum(torch.sum(p ** 2) for p in self.model.parameters()) / (2 * self.prior_std ** 2)
                loss += prior_loss
                
                loss.backward()
                
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise_std = np.sqrt(2 * learning_rate * self.temperature / len(train_loader))
                        param.data -= learning_rate * param.grad + torch.randn_like(param) * noise_std
                
                self.model.zero_grad()
                
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                total_entropy += entropy.item()
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(y_train.cpu().numpy(), predictions)
            train_entropy = total_entropy / len(train_loader)
            
            val_metrics = self.evaluate(X_val, y_val)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_entropy'].append(train_entropy)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_entropy'].append(val_metrics['entropy'])
            self.history['temperature'].append(self.temperature)
            
            # Log metrics every 100 epochs
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Temp: {self.temperature:.4f}")
            
            if epoch >= burn_in_epochs and (epoch - burn_in_epochs) % sampling_interval == 0:
                self._collect_sample()
                print(f"Epoch {epoch+1}: Collected model sample ({len(self.samples)}/{num_ensemble_models})")
    
    def evaluate(self, X, y):
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
            'entropy': entropy
        }
    
    def _collect_sample(self):
        self.samples.append({'state_dict': {k: v.clone().cpu() for k, v in self.model.state_dict().items()}})
    
    def get_models(self):
        return self.samples, self.history 