# File: tests/test_text_classifier.py

import pytest
import numpy as np
from src.models.text_classifier import TextClassifier

@pytest.fixture
def sample_data():
    """Fixture providing sample text data and labels."""
    texts = [
        "This is a positive happy review!",
        "Terrible experience, very disappointing",
        "Great product, highly recommend",
        "Waste of money, do not buy",
        "Absolutely love it, perfect purchase"
    ]
    labels = ["positive", "negative", "positive", "negative", "positive"]
    return texts, labels

def test_classifier_initialization():
    """Test classifier initialization."""
    classifier = TextClassifier()
    assert classifier.model_name == "text_classifier"
    assert not classifier.trained
    assert classifier.config['model_type'] == "logistic"

def test_classifier_training(sample_data):
    """Test model training and prediction."""
    texts, labels = sample_data
    
    # Initialize and train classifier
    classifier = TextClassifier()
    metrics = classifier.train(texts, labels)
    
    # Check training results
    assert classifier.trained
    assert isinstance(metrics, dict)
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics
    
    # Test predictions
    predictions = classifier.predict(texts)
    assert len(predictions) == len(texts)
    assert all(pred in classifier.classes_ for pred in predictions)
    
    # Test probability predictions
    probas = classifier.predict_proba(texts)
    assert probas.shape == (len(texts), len(classifier.classes_))
    assert np.allclose(np.sum(probas, axis=1), 1.0)

def test_feature_importance(sample_data):
    """Test feature importance extraction."""
    texts, labels = sample_data
    
    classifier = TextClassifier()
    classifier.train(texts, labels)
    
    importance_df = classifier.get_feature_importance(top_n=5)
    assert len(importance_df) == 5 * len(classifier.classes_)
    assert all(col in importance_df.columns for col in ['feature', 'importance', 'class'])

def test_model_persistence(sample_data, tmp_path):
    """Test model saving and loading."""
    texts, labels = sample_data
    
    # Train and save model
    classifier = TextClassifier()
    classifier.train(texts, labels)
    save_dir = tmp_path / "model"
    classifier.save_model(str(save_dir))
    
    # Load model and make predictions
    new_classifier = TextClassifier()
    new_classifier.load_model(str(save_dir))
    
    # Verify predictions match
    assert np.array_equal(
        classifier.predict(texts),
        new_classifier.predict(texts)
    )