# File: src/models/base_model.py

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import joblib
import json
import os

class BaseTextModel(ABC):
    """Abstract base class for all text models in the project."""
    
    def __init__(self, model_name: str):
        """
        Initialize base text model.
        
        Args:
            model_name (str): Name identifier for the model
        """
        self.model_name = model_name
        self.model: Optional[BaseEstimator] = None
        self.trained = False
        self.config: Dict[str, Any] = {}
        
    @abstractmethod
    def train(self, X: np.ndarray, y: Any) -> None:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (Any): Target labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    def save_model(self, directory: str) -> None:
        """
        Save model and configuration to disk.
        
        Args:
            directory (str): Directory to save model files
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        # Save configuration
        config_path = os.path.join(directory, f"{self.model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
    
    def load_model(self, directory: str) -> None:
        """
        Load model and configuration from disk.
        
        Args:
            directory (str): Directory containing model files
        """
        # Load model
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        self.model = joblib.load(model_path)
        
        # Load configuration
        config_path = os.path.join(directory, f"{self.model_name}_config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.trained = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and configuration.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        return {
            'model_name': self.model_name,
            'trained': self.trained,
            'config': self.config,
            'model_type': type(self.model).__name__ if self.model else None
        }