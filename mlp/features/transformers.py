"""
Custom feature transformers for text data.

This module provides scikit-learn compatible transformers for extracting
various features from text data, particularly useful for sentiment analysis
and other NLP tasks.
"""

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceCounter(BaseEstimator, TransformerMixin):
    """
    Extract the number of sentences in each text sample.

    Counts sentences by looking for sentence-ending punctuation (., !, ?).

    Example:
        >>> counter = SentenceCounter()
        >>> features = counter.fit_transform(["Hello. World!", "Test?"])
        >>> print(features)
        [[2.]
         [1.]]
    """

    def fit(self, data, y=None):
        """
        Fit method (does nothing, included for sklearn compatibility).

        Args:
            data: Array-like of text samples
            y: Ignored, present for API compatibility

        Returns:
            self
        """
        return self

    def transform(self, data):
        """
        Count sentences in each text sample.

        Args:
            data: Array-like of text samples

        Returns:
            Feature array of shape (n_samples, 1) with sentence counts
        """
        features = np.zeros((len(data), 1))

        for i, review in enumerate(data):
            sentence_count = (review.count('.') +
                            review.count('!') +
                            review.count('?'))
            features[i, 0] = sentence_count

        return features


class PunctCounter(SentenceCounter):
    """
    Extract the number of exclamation marks and question marks.

    Useful for detecting emotional or interrogative content in text.

    Example:
        >>> counter = PunctCounter()
        >>> features = counter.fit_transform(["Hello! World!", "Test?"])
        >>> print(features)
        [[2.]
         [1.]]
    """

    def transform(self, data):
        """
        Count exclamation marks and question marks in each text sample.

        Args:
            data: Array-like of text samples

        Returns:
            Feature array of shape (n_samples, 1) with punctuation counts
        """
        features = np.zeros((len(data), 1))

        for i, review in enumerate(data):
            punct_count = review.count('!') + review.count('?')
            features[i, 0] = punct_count

        return features


class NegativeCounter(SentenceCounter):
    """
    Count occurrences of negative words in text.

    Uses a predefined vocabulary of negative sentiment words to
    count negative content in each text sample.

    Attributes:
        neg_words (list): Vocabulary of negative sentiment words
    """

    def __init__(self):
        """Initialize with predefined negative word vocabulary."""
        self.neg_words = [
            "second-rate", "violent", "moronic", "third-rate", "flawed",
            "juvenile", "boring", "distasteful", "ordinary", "disgusting",
            "senseless", "static", "brutal", "confused", "disappointing",
            "bloody", "silly", "tired", "predictable", "stupid",
            "uninteresting", "weak", "incredibly tiresome", "trite",
            "uneven", "cliché ridden", "outdated", "dreadful", "bland",
            "bad", "worst", "waste"
        ]

    def transform(self, data):
        """
        Count negative words in each text sample.

        Handles both single words and multi-word phrases in the vocabulary.

        Args:
            data: Array-like of text samples

        Returns:
            Feature array of shape (n_samples, 1) with negative word counts
        """
        features = np.zeros((len(data), 1))

        for i, review in enumerate(data):
            tokens = word_tokenize(review.lower())
            count = 0
            review_lower = review.lower()

            for neg_word in self.neg_words:
                if ' ' in neg_word:
                    # Multi-word phrase: search in full text
                    count += review_lower.count(neg_word)
                else:
                    # Single word: search in tokens
                    count += tokens.count(neg_word)

            features[i, 0] = count

        return features


class PositiveCounter(NegativeCounter):
    """
    Count occurrences of positive words in text.

    Uses a predefined vocabulary of positive sentiment words to
    count positive content in each text sample.

    Attributes:
        pos_words (list): Vocabulary of positive sentiment words
    """

    def __init__(self):
        """Initialize with predefined positive word vocabulary."""
        self.pos_words = [
            "first-rate", "insightful", "clever", "charming", "comical",
            "charismatic", "enjoyable", "uproarious", "original", "tender",
            "hilarious", "absorbing", "sensitive", "riveting", "intriguing",
            "powerful", "fascinating", "pleasant", "surprising", "dazzling",
            "imaginative", "legendary", "unpretentious", "love", "wonderful",
            "best", "great", "superb", "still", "beautiful"
        ]

    def transform(self, data):
        """
        Count positive words in each text sample.

        Handles both single words and multi-word phrases in the vocabulary.

        Args:
            data: Array-like of text samples

        Returns:
            Feature array of shape (n_samples, 1) with positive word counts
        """
        features = np.zeros((len(data), 1))

        for i, review in enumerate(data):
            tokens = word_tokenize(review.lower())
            count = 0
            review_lower = review.lower()

            for pos_word in self.pos_words:
                if ' ' in pos_word:
                    # Multi-word phrase: search in full text
                    count += review_lower.count(pos_word)
                else:
                    # Single word: search in tokens
                    count += tokens.count(pos_word)

            features[i, 0] = count

        return features
