"""
Feature engineering utilities for text processing.

This package provides scikit-learn compatible transformers for extracting
features from text data, including sentence counting and sentiment word counting.
"""

from .transformers import (
    SentenceCounter,
    PunctCounter,
    NegativeCounter,
    PositiveCounter
)

__all__ = [
    'SentenceCounter',
    'PunctCounter',
    'NegativeCounter',
    'PositiveCounter',
]
