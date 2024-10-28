from typing import List, Dict, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class TextFeatureExtractor:
    """Extract features from cleaned text for ML models."""
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize the feature extractor.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        
    def get_basic_stats(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get basic statistical features from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of text statistics
        """
        words = text.split()
        word_lengths = [len(word) for word in words]
        
        return {
            'word_count': len(words),
            'avg_word_length': np.mean(word_lengths) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'char_count': len(text),
            'density': len(text) / (len(words) if words else 1)
        }
    
    def extract_tfidf(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts (List[str]): List of texts to extract features from
            fit (bool): Whether to fit the vectorizer or use existing fit
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if fit or not self.fitted:
            features = self.tfidf.fit_transform(texts)
            self.fitted = True
        else:
            features = self.tfidf.transform(texts)
            
        return features.toarray()
    
    def get_top_terms(self, text: str, n: int = 10) -> List[tuple]:
        """
        Get top n most frequent terms in text.
        
        Args:
            text (str): Input text
            n (int): Number of top terms to return
            
        Returns:
            List[tuple]: List of (term, frequency) pairs
        """
        words = text.split()
        counter = Counter(words)
        return counter.most_common(n)