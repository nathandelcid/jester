"""
Example: IMDB sentiment analysis with feature engineering.

This script demonstrates how to:
1. Load the IMDB dataset
2. Engineer features from text using custom transformers
3. Train a scikit-learn classifier with the extracted features
4. Evaluate performance
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import IMDB
from features import SentenceCounter, PunctCounter, NegativeCounter, PositiveCounter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


def simple_features_demo():
    """Demo with simple counting features."""
    print("=" * 60)
    print("IMDB Sentiment Analysis - Simple Features")
    print("=" * 60)

    # Load dataset
    print("\nNote: This example requires movie_review_data.json")
    print("Please ensure the data file is in the correct location.")
    try:
        imdb = IMDB()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print(f"Dataset loaded: {len(imdb.X_train)} training samples, "
          f"{len(imdb.X_test)} test samples")

    # Extract features
    print("\nExtracting features: sentence count, negative words, positive words")
    featurizer = FeatureUnion([
        ('sentences_count', SentenceCounter()),
        ('negative_words_count', NegativeCounter()),
        ('positive_words_count', PositiveCounter())
    ])

    X_train = featurizer.fit_transform(imdb.X_train)
    X_test = featurizer.transform(imdb.X_test)

    print(f"Feature shape: {X_train.shape}")

    # Train classifier
    print("\nTraining logistic regression classifier...")
    lr = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001,
                      max_iter=2000, shuffle=True, verbose=0, random_state=42)
    lr.fit(X_train, imdb.y_train)

    # Evaluate
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_pred, imdb.y_test)

    print(f"\nAccuracy on test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(imdb.y_test, y_pred,
                               target_names=['Negative', 'Positive']))


def ngram_features_demo():
    """Demo with n-gram features."""
    print("\n" + "=" * 60)
    print("IMDB Sentiment Analysis - N-gram Features")
    print("=" * 60)

    # Load dataset
    try:
        imdb = IMDB()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print(f"Dataset loaded: {len(imdb.X_train)} training samples, "
          f"{len(imdb.X_test)} test samples")

    # Extract features with n-grams
    print("\nExtracting features: sentence count, sentiment words, and word n-grams")
    featurizer = FeatureUnion([
        ('sentences_count', SentenceCounter()),
        ('negative_words_count', NegativeCounter()),
        ('positive_words_count', PositiveCounter()),
        ('word_ngrams', CountVectorizer(ngram_range=(1, 3), max_features=5000))
    ])

    X_train = featurizer.fit_transform(imdb.X_train)
    X_test = featurizer.transform(imdb.X_test)

    print(f"Feature shape: {X_train.shape}")

    # Train classifier
    print("\nTraining logistic regression classifier...")
    lr = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001,
                      max_iter=2000, shuffle=True, verbose=0, random_state=42)
    lr.fit(X_train, imdb.y_train)

    # Evaluate
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_pred, imdb.y_test)

    print(f"\nAccuracy on test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(imdb.y_test, y_pred,
                               target_names=['Negative', 'Positive']))

    # Show some example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    for i in range(3):
        text_preview = imdb.X_test[i][:100] + "..."
        true_label = "Positive" if imdb.y_test[i] == 1 else "Negative"
        pred_label = "Positive" if y_pred[i] == 1 else "Negative"
        correct = "✓" if imdb.y_test[i] == y_pred[i] else "✗"

        print(f"\nExample {i+1} {correct}")
        print(f"Text: {text_preview}")
        print(f"True: {true_label}, Predicted: {pred_label}")


if __name__ == "__main__":
    # Run simple features demo
    simple_features_demo()

    # Run n-gram features demo
    ngram_features_demo()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
