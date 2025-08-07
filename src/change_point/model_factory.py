"""
Model Factory for Change-Point Detection.

Factory pattern implementation for creating different types of change-point models
with standardized interfaces and configuration management.
"""

from typing import Dict, Any, Optional, Type, List
from loguru import logger
import yaml
from pathlib import Path

from .bayesian_detector import BayesianChangePointDetector


class ModelFactory:
    """
    Factory for creating change-point detection models.
    
    Supports multiple model types and configurations following
    the factory design pattern for extensibility.
    """
    
    _model_registry: Dict[str, Type] = {
        'bayesian': BayesianChangePointDetector,
        'bayesian_basic': BayesianChangePointDetector,
        'bayesian_robust': BayesianChangePointDetector,
        'bayesian_hierarchical': BayesianChangePointDetector,
        'bayesian_online': BayesianChangePointDetector,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a change-point detection model.
        
        Args:
            model_type: Type of model to create
            config: Optional configuration dictionary
            
        Returns:
            Initialized model instance
        """
        if model_type not in cls._model_registry:
            available_types = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available: {available_types}")
        
        model_class = cls._model_registry[model_type]
        
        # Set model-specific configuration
        if config is None:
            config = cls._get_default_config(model_type)
        else:
            # Merge with defaults
            default_config = cls._get_default_config(model_type)
            config = cls._merge_configs(default_config, config)
        
        # Set the specific model architecture
        if model_type.startswith('bayesian_'):
            architecture = model_type.replace('bayesian_', '')
            if architecture in ['basic', 'robust', 'hierarchical', 'online']:
                config['model_type'] = architecture
        
        logger.info(f"Creating {model_type} model with config: {config.get('model_type', 'default')}")
        
        return model_class(config=config)
    
    @classmethod
    def _get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type."""
        base_config = {
            'max_changepoints': 25,
            'changepoint_prior_scale': 0.05,
            'changepoint_range': 0.8,
            'sigma_prior': 5.0,
            'mu_prior_scale': 10.0,
            'mcmc': {
                'draws': 2000,
                'tune': 1000,
                'chains': 4,
                'cores': 4,
                'target_accept': 0.95,
                'max_treedepth': 12
            }
        }
        
        # Model-specific configurations
        model_configs = {
            'bayesian_basic': {
                'model_type': 'basic',
                'max_changepoints': 15,
            },
            'bayesian_robust': {
                'model_type': 'robust',
                'nu_prior': 4.0,
                'max_changepoints': 20,
            },
            'bayesian_hierarchical': {
                'model_type': 'hierarchical',
                'max_changepoints': 25,
                'hierarchical_prior': True,
            },
            'bayesian_online': {
                'model_type': 'online',
                'max_changepoints': 10,
                'online_threshold': 0.7,
            }
        }
        
        if model_type in model_configs:
            return cls._merge_configs(base_config, model_configs[model_type])
        
        return base_config
    
    @classmethod
    def _merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def load_config_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    @classmethod
    def create_from_config_file(cls, config_path: str, model_type: Optional[str] = None) -> Any:
        """
        Create model from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            model_type: Optional override for model type
            
        Returns:
            Initialized model instance
        """
        config = cls.load_config_from_file(config_path)
        
        # Extract model configuration
        model_config = config.get('change_point_model', {})
        
        # Override model type if specified
        if model_type is not None:
            model_config['model_type'] = model_type
        
        # Get model type from config
        model_type = model_config.get('model_type', 'bayesian_hierarchical')
        
        return cls.create_model(model_type, model_config)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name for the model type
            model_class: Model class to register
        """
        cls._model_registry[name] = model_class
        logger.info(f"Registered new model type: {name}")
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types."""
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        default_config = cls._get_default_config(model_type)
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'description': model_class.__doc__ or "No description available",
            'default_config': default_config
        }
