# File: src/models/text_classifier.py

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from .base_model import BaseTextModel
from ..features.extractors.text_features import TextFeatureExtractor
from ..data.preprocessors.text_cleaner import TextCleaner

class TextClassifier(BaseTextModel):
    """Text classification model with preprocessing pipeline."""
    
    def __init__(
        self,
        model_name: str = "text_classifier",
        max_features: int = 5000,
        model_type: str = "logistic",
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize text classifier.
        
        Args:
            model_name (str): Name identifier for the model
            max_features (int): Maximum number of features for TF-IDF
            model_type (str): Type of model to use ('logistic' for now)
            model_params (Optional[Dict]): Parameters for the classifier
        """
        super().__init__(model_name)
        
        self.cleaner = TextCleaner()
        self.feature_extractor = TextFeatureExtractor(max_features=max_features)
        
        if model_type == "logistic":
            default_params = {
                'C': 1.0,
                'max_iter': 1000,
                'multi_class': 'auto',
                'random_state': 42
            }
            if model_params:
                default_params.update(model_params)
            self.model = LogisticRegression(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.config = {
            'model_type': model_type,
            'max_features': max_features,
            'model_params': default_params
        }
        
        self.classes_ = None
        self.feature_names_ = None
    
    def train(self, texts: List[str], labels: List[Any]) -> Dict[str, Any]:
        """
        Train the model on text data.
        
        Args:
            texts (List[str]): List of training texts
            labels (List[Any]): Target labels
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        # Preprocess texts
        cleaned_texts = self.cleaner.clean_texts(texts)
        
        # Extract features
        X = self.feature_extractor.extract_tfidf(cleaned_texts, fit=True)
        
        # Store feature names
        self.feature_names_ = self.feature_extractor.tfidf.get_feature_names_out()
        
        # Train model
        self.model.fit(X, labels)
        
        # Store classes
        self.classes_ = self.model.classes_
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(labels, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        self.trained = True
        return metrics
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on new texts.
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.trained:
            raise ValueError("Model must be trained before predicting")
            
        # Preprocess texts
        cleaned_texts = self.cleaner.clean_texts(texts)
        
        # Extract features
        X = self.feature_extractor.extract_tfidf(cleaned_texts, fit=False)
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for texts.
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            np.ndarray: Prediction probabilities for each class
        """
        if not self.trained:
            raise ValueError("Model must be trained before predicting")
            
        cleaned_texts = self.cleaner.clean_texts(texts)
        X = self.feature_extractor.extract_tfidf(cleaned_texts, fit=False)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get the most important features for each class.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: DataFrame with feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
            
        # Get coefficients from the model
        coefficients = self.model.coef_
        
        # Create DataFrame for each class
        importance_dfs = []
        for i, class_label in enumerate(self.classes_):
            class_coeffs = coefficients[i] if len(self.classes_) > 2 else coefficients[0]
            
            # Create DataFrame with feature names and coefficients
            class_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance': np.abs(class_coeffs)
            })
            
            # Sort by absolute importance and get top N
            class_df = class_df.nlargest(top_n, 'importance')
            class_df['class'] = class_label
            
            importance_dfs.append(class_df)
        
        # Combine all classes
        return pd.concat(importance_dfs, ignore_index=True)