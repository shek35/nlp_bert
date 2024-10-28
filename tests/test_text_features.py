import numpy as np
from src.features.extractors.text_features import TextFeatureExtractor
from src.data.preprocessors.text_cleaner import TextCleaner

def test_feature_extractor():
    """Test the TextFeatureExtractor functionality."""
    
    # Initialize our classes
    cleaner = TextCleaner()
    extractor = TextFeatureExtractor(max_features=100)
    
    # Sample texts
    sample_texts = [
        "Machine learning is an amazing field of computer science!",
        "Data science combines statistics and programming.",
        "Natural language processing helps computers understand text.",
        "Python is a popular programming language for AI."
    ]
    
    # Clean the texts first
    cleaned_texts = cleaner.clean_texts(sample_texts)
    
    print("Testing Basic Statistics:")
    print("-" * 50)
    # Test basic statistics
    for text in cleaned_texts:
        stats = extractor.get_basic_stats(text)
        print(f"\nText: {text}")
        print("Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
    
    print("\nTesting TF-IDF Extraction:")
    print("-" * 50)
    # Test TF-IDF features
    tfidf_features = extractor.extract_tfidf(cleaned_texts)
    print(f"\nTF-IDF Matrix Shape: {tfidf_features.shape}")
    print(f"Number of features: {tfidf_features.shape[1]}")
    
    print("\nTesting Top Terms:")
    print("-" * 50)
    # Test top terms extraction
    for i, text in enumerate(cleaned_texts, 1):
        print(f"\nText {i} top terms:")
        top_terms = extractor.get_top_terms(text, n=3)
        for term, freq in top_terms:
            print(f"'{term}': {freq} occurrences")

if __name__ == "__main__":
    test_feature_extractor()