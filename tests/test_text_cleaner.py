def test_text_cleaner():
    from src.data.preprocessors.text_cleaner import TextCleaner
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Test case 1: Basic cleaning
    test_text = "Hello! This is a test text with numbers 123 and URL: https://example.com"
    cleaned = cleaner.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print()
    
    # Test case 2: Multiple texts
    texts = [
        "First text with email@example.com",
        "Second text with numbers 456",
        "Third text with URL www.example.com"
    ]
    cleaned_texts = cleaner.clean_texts(texts)
    for orig, clean in zip(texts, cleaned_texts):
        print(f"Original: {orig}")
        print(f"Cleaned: {clean}")
        print()
    
    # Test case 3: Get statistics
    stats = cleaner.get_cleaning_stats(test_text, cleaned)
    print("Cleaning Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_text_cleaner()