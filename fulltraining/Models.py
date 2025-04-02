import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import logging
from decorators import log_class_methods, LogLevel

# Set up logging
logger = logging.getLogger(__name__)

@log_class_methods(default_log_level=LogLevel.INFO)
class BaseNN(nn.Module):
    """Base class for neural networks with common functionality."""
    
    def __init__(self, input_dim: int):
        """Initialize the base neural network with input dimension validation."""
        if input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")
        
        super(BaseNN, self).__init__()
        self.input_dim = input_dim

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

@log_class_methods(default_log_level=LogLevel.INFO)
class SimpleNN(BaseNN):
    """Simple neural network with dropout and batch normalization."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """Initialize a simple neural network with configurable architecture."""
        if not all(dim > 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        super(SimpleNN, self).__init__(input_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer for 10 classes
        layers.append(nn.Linear(prev_dim, 10))
        layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network with input validation."""
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        return self.layers(x)

@log_class_methods(default_log_level=LogLevel.INFO)
class BayesianNN(BaseNN):
    """Bayesian neural network with proper prior distributions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        prior_std: float = 0.1
    ):
        """Initialize a Bayesian neural network with configurable architecture and prior."""
        if not all(dim > 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if prior_std <= 0:
            raise ValueError("Prior standard deviation must be positive")
        
        super(BayesianNN, self).__init__(input_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.prior_std = prior_std
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer for 10 classes
        layers.append(nn.Linear(prev_dim, 10))
        layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize parameters from prior
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0, self.prior_std)
            if module.bias is not None:
                module.bias.data.normal_(0, self.prior_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network with input validation."""
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        return self.layers(x)

    def log_prior(self) -> torch.Tensor:
        """Compute log prior (L2 regularization)."""
        log_prior = 0.0
        for param in self.parameters():
            log_prior += torch.sum(param ** 2) / (2 * self.prior_std ** 2)
        return log_prior