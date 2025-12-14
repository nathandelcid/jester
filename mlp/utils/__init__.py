"""
Utility functions and classes for data loading and visualization.

This package provides dataset loaders (Circles, IMDB) and visualization
utilities for neural network models.
"""

from .data import Circles, IMDB
from .visualization import show_decision_surface

__all__ = [
    'Circles',
    'IMDB',
    'show_decision_surface',
]
