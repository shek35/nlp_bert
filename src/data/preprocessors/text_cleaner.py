import re
import unicodedata
from typing import List, Optional

import nltk
from nltk.corpus import stopwords

class TextCleaner:
    """Text preprocessing and cleaning class for NLP tasks."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the text cleaner.
        
        Args:
            language (str): Language for stopwords. Defaults to 'english'.
        """
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words(language))
        self.language = language
    
    def clean_text(self, text: str,
                  remove_stopwords: bool = True,
                  remove_numbers: bool = True,
                  remove_punctuation: bool = True,
                  min_word_length: int = 2) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text (str): Input text to clean
            remove_stopwords (bool): Whether to remove stopwords
            remove_numbers (bool): Whether to remove numbers
            remove_punctuation (bool): Whether to remove punctuation
            min_word_length (int): Minimum word length to keep
        
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers if specified
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        
        # Clean tokens
        tokens = [
            token for token in tokens 
            if (
                len(token) >= min_word_length and 
                (token not in self.stop_words if remove_stopwords else True)
            )
        ]
        
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_texts(self, texts: List[str], **kwargs) -> List[str]:
        """
        Clean a list of texts.
        
        Args:
            texts (List[str]): List of texts to clean
            **kwargs: Arguments to pass to clean_text
            
        Returns:
            List[str]: List of cleaned texts
        """
        return [self.clean_text(text, **kwargs) for text in texts]

    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> dict:
        """
        Get statistics about the cleaning process.
        
        Args:
            original_text (str): Original text before cleaning
            cleaned_text (str): Text after cleaning
            
        Returns:
            dict: Dictionary containing cleaning statistics
        """
        original_tokens = original_text.lower().split()
        cleaned_tokens = cleaned_text.split()
        
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'original_tokens': len(original_tokens),
            'cleaned_tokens': len(cleaned_tokens),
            'reduction_percentage': round(
                (1 - len(cleaned_text) / len(original_text)) * 100 
                if len(original_text) > 0 else 0,
                2
            )
        }