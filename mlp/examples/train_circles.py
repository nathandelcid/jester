"""
Example: Training an MLP on the concentric circles dataset.

This script demonstrates how to:
1. Load the circles dataset
2. Build a multi-layer perceptron
3. Train the network
4. Visualize the decision surface
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import Network, Dense, ReLU, Sigmoid, MSE, SGD, Adam
from utils import Circles, show_decision_surface


def train_circles_sgd():
    """Train MLP on circles dataset using SGD optimizer."""
    print("=" * 60)
    print("Training MLP on Circles Dataset with SGD")
    print("=" * 60)

    # Load dataset
    circles = Circles()
    print(f"Dataset loaded: {len(circles.X_train)} training samples, "
          f"{len(circles.X_test)} test samples")

    # Build network
    network = Network(optimizer=SGD(learning_rate=0.5), loss=MSE())
    network.add_layer(Dense(2, 16))
    network.add_layer(ReLU())
    network.add_layer(Dense(16, 8))
    network.add_layer(ReLU())
    network.add_layer(Dense(8, 1))
    network.add_layer(Sigmoid())

    print(f"Network architecture: 2 -> 16 -> 8 -> 1")
    print(f"Optimizer: SGD (learning_rate=0.5)")

    # Training parameters
    batch_size = 10
    epochs = 100
    losses = []

    print(f"\nTraining for {epochs} epochs with batch_size={batch_size}...")

    # Training loop
    for epoch in range(epochs):
        batch_idx = list(range(0, len(circles.X_train), batch_size))
        np.random.shuffle(batch_idx)

        for i in batch_idx:
            batch_x = circles.X_train[i:i+batch_size]
            batch_y = circles.y_train[i:i+batch_size].reshape(-1, 1)
            _ = network.fit(batch_x, batch_y)

        # Evaluate on full dataset
        y_pred = network.predict(circles.X)
        loss = network.loss.forward(y_pred, circles.labels.reshape(-1, 1))
        losses.append(loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")

    # Plot training loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss (SGD)')
    plt.grid(True, alpha=0.3)

    # Visualize decision surface
    plt.subplot(1, 2, 2)
    show_decision_surface(network, circles.X, circles.labels, ax=plt.gca())

    plt.tight_layout()
    plt.savefig('circles_sgd_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to circles_sgd_results.png")
    plt.show()


def train_circles_adam():
    """Train MLP on circles dataset using Adam optimizer."""
    print("\n" + "=" * 60)
    print("Training MLP on Circles Dataset with Adam")
    print("=" * 60)

    # Load dataset
    circles = Circles()
    print(f"Dataset loaded: {len(circles.X_train)} training samples, "
          f"{len(circles.X_test)} test samples")

    # Build network
    network = Network(optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
                     loss=MSE())
    network.add_layer(Dense(2, 16))
    network.add_layer(ReLU())
    network.add_layer(Dense(16, 8))
    network.add_layer(ReLU())
    network.add_layer(Dense(8, 1))
    network.add_layer(Sigmoid())

    print(f"Network architecture: 2 -> 16 -> 8 -> 1")
    print(f"Optimizer: Adam (learning_rate=0.01, beta_1=0.9, beta_2=0.999)")

    # Training parameters
    batch_size = 10
    epochs = 100
    losses = []

    print(f"\nTraining for {epochs} epochs with batch_size={batch_size}...")

    # Training loop
    for epoch in range(epochs):
        batch_idx = list(range(0, len(circles.X_train), batch_size))
        np.random.shuffle(batch_idx)

        for i in batch_idx:
            batch_x = circles.X_train[i:i+batch_size]
            batch_y = circles.y_train[i:i+batch_size].reshape(-1, 1)
            _ = network.fit(batch_x, batch_y)

        # Evaluate on full dataset
        y_pred = network.predict(circles.X)
        loss = network.loss.forward(y_pred, circles.labels.reshape(-1, 1))
        losses.append(loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")

    # Plot training loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss (Adam)')
    plt.grid(True, alpha=0.3)

    # Visualize decision surface
    plt.subplot(1, 2, 2)
    show_decision_surface(network, circles.X, circles.labels, ax=plt.gca())

    plt.tight_layout()
    plt.savefig('circles_adam_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to circles_adam_results.png")
    plt.show()


if __name__ == "__main__":
    # Train with SGD
    train_circles_sgd()

    # Train with Adam
    train_circles_adam()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
