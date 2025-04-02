import yaml
from pathlib import Path
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the configuration manager."""
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.config['data']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training parameters configuration."""
        return self.config['training']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config['evaluation']
    
    def get_random_seed(self) -> int:
        """Get random seed."""
        return self.config['random_seed']
    
    def get_device(self) -> str:
        """Get device configuration."""
        return self.config['evaluation']['device']
    
    def get_simplenn_config(self) -> Dict[str, Any]:
        """Get SimpleNN training configuration."""
        return self.config['training']['simplenn']
    
    def get_sgld_config(self) -> Dict[str, Any]:
        """Get SGLD training configuration."""
        return self.config['training']['sgld'] 