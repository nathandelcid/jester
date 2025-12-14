"""
Visualization utilities for neural network models.

This module provides functions for visualizing model predictions and
decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_decision_surface(model, X, y, ax=None):
    """
    Visualize the decision surface of a trained classification model.

    Creates a contour plot showing the model's predicted probabilities
    across the feature space, overlaid with the actual data points.

    Args:
        model: Trained model with a predict() method
        X: Feature matrix of shape (n_samples, 2) - must be 2D for visualization
        y: True labels of shape (n_samples,)
        ax: Optional matplotlib axis to plot on. If None, creates new figure.

    Returns:
        None (displays plot)

    Example:
        >>> from mlp import Network, Dense, ReLU, Sigmoid, MSE, SGD
        >>> from utils.data import Circles
        >>> from utils.visualization import show_decision_surface
        >>>
        >>> circles = Circles()
        >>> network = Network(optimizer=SGD(0.1), loss=MSE())
        >>> # ... train network ...
        >>> show_decision_surface(network, circles.X, circles.labels)
    """
    # Define grid boundaries with padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create meshgrid for predictions
    x_grid, y_grid = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Prepare grid points for prediction
    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))

    # Get predictions
    y_hat = model.predict(grid).reshape(-1,)
    zz = y_hat.reshape(xx.shape)

    # Plot decision surface
    if ax is None:
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, cmap='PiYG', alpha=0.8)
        plt.colorbar(label='Predicted probability')
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='PiYG')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Surface')
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='PiYG')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Surface')
