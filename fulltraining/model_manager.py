import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging
from Models import SimpleNN, BayesianNN

# Set up logging
logger = logging.getLogger(__name__)

def ensure_models_dir() -> Path:
    """Create and return the directory for storing trained models."""
    current_dir = Path(__file__).parent
    models_dir = current_dir / "trained_models"
    models_dir.mkdir(exist_ok=True)
    return models_dir

def save_model(model: torch.nn.Module, name: str) -> None:
    """Save a model to disk."""
    try:
        models_dir = ensure_models_dir()
        save_path = models_dir / f"{name}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(model: torch.nn.Module, name: str, device: torch.device) -> torch.nn.Module:
    """Load a model from disk."""
    try:
        models_dir = ensure_models_dir()
        load_path = models_dir / f"{name}.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model.load_state_dict(torch.load(load_path))
        model = model.to(device)
        logger.info(f"Loaded model from {load_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_sgld_samples(samples: List[Dict[str, torch.Tensor]], name: str) -> None:
    """Save SGLD posterior samples to disk."""
    try:
        models_dir = ensure_models_dir()
        save_path = models_dir / f"{name}_samples.pt"
        torch.save(samples, save_path)
        logger.info(f"Saved SGLD samples to {save_path}")
    except Exception as e:
        logger.error(f"Error saving SGLD samples: {e}")
        raise

def load_sgld_samples(name: str) -> List[Dict[str, torch.Tensor]]:
    """Load SGLD posterior samples from disk."""
    try:
        models_dir = ensure_models_dir()
        load_path = models_dir / f"{name}_samples.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"Samples file not found: {load_path}")
        
        samples = torch.load(load_path)
        logger.info(f"Loaded SGLD samples from {load_path}")
        return samples
    except Exception as e:
        logger.error(f"Error loading SGLD samples: {e}")
        raise 