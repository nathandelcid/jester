# MLP from Scratch

A minimal neural network framework for building Multi-Layer Perceptrons (MLPs) from scratch using only NumPy. This project implements forward and backward propagation, various layer types, loss functions, and optimization algorithms without relying on deep learning frameworks like PyTorch or TensorFlow.

## Features

- **Core Neural Network Components**
  - Dense (fully connected) layers with automatic gradient computation
  - Activation functions: Sigmoid, ReLU
  - Loss functions: MSE (Mean Squared Error)
  - Parameter tracking with gradient accumulation

- **Optimization Algorithms**
  - SGD (Stochastic Gradient Descent)
  - Adam (Adaptive Moment Estimation)

- **Feature Engineering Tools**
  - Text feature extractors for NLP tasks
  - Scikit-learn compatible transformers
  - Sentence and sentiment word counters

- **Utilities**
  - Dataset loaders (Circles, IMDB)
  - Decision surface visualization
  - Training examples

## Installation

### From source

```bash
cd mlp_from_scratch
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- Scikit-learn >= 0.24.0
- NLTK >= 3.6.0

## Quick Start

### Building a Simple MLP

```python
from mlp import Network, Dense, ReLU, Sigmoid, MSE, SGD

# Create network
network = Network(optimizer=SGD(learning_rate=0.1), loss=MSE())

# Add layers
network.add_layer(Dense(2, 16))   # Input: 2 features, Output: 16
network.add_layer(ReLU())          # ReLU activation
network.add_layer(Dense(16, 8))   # Hidden layer
network.add_layer(ReLU())
network.add_layer(Dense(8, 1))    # Output layer
network.add_layer(Sigmoid())       # Sigmoid activation for binary classification

# Train
loss = network.fit(X_train, y_train)

# Predict
predictions = network.predict(X_test)
```

### Training on Circles Dataset

```python
from mlp import Network, Dense, ReLU, Sigmoid, MSE, Adam
from utils import Circles, show_decision_surface
import numpy as np

# Load dataset
circles = Circles()

# Build network
network = Network(optimizer=Adam(learning_rate=0.01), loss=MSE())
network.add_layer(Dense(2, 16))
network.add_layer(ReLU())
network.add_layer(Dense(16, 8))
network.add_layer(ReLU())
network.add_layer(Dense(8, 1))
network.add_layer(Sigmoid())

# Training loop
batch_size = 10
epochs = 100

for epoch in range(epochs):
    batch_idx = list(range(0, len(circles.X_train), batch_size))
    np.random.shuffle(batch_idx)

    for i in batch_idx:
        batch_x = circles.X_train[i:i+batch_size]
        batch_y = circles.y_train[i:i+batch_size].reshape(-1, 1)
        loss = network.fit(batch_x, batch_y)

# Visualize
show_decision_surface(network, circles.X, circles.labels)
```

### Feature Engineering for Text

```python
from features import SentenceCounter, NegativeCounter, PositiveCounter
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier

# Create feature pipeline
featurizer = FeatureUnion([
    ('sentences_count', SentenceCounter()),
    ('negative_words_count', NegativeCounter()),
    ('positive_words_count', PositiveCounter())
])

# Extract features
X_train = featurizer.fit_transform(texts_train)
X_test = featurizer.transform(texts_test)

# Train classifier
classifier = SGDClassifier(loss='log_loss')
classifier.fit(X_train, y_train)
```

## Project Structure

```
mlp_from_scratch/
├── mlp/                    # Core neural network framework
│   ├── __init__.py
│   ├── parameter.py        # Parameter class with gradient tracking
│   ├── layers.py           # Layer implementations (Dense, Sigmoid, ReLU)
│   ├── losses.py           # Loss functions (MSE)
│   ├── optimizers.py       # Optimization algorithms (SGD, Adam)
│   └── network.py          # Network container class
├── features/               # Feature engineering tools
│   ├── __init__.py
│   └── transformers.py     # Text feature extractors
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── data.py            # Dataset loaders
│   └── visualization.py   # Plotting functions
├── examples/               # Example scripts
│   ├── train_circles.py   # Circles dataset example
│   └── imdb_sentiment.py  # IMDB sentiment analysis
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── setup.py               # Package installation
└── README.md              # This file
```

## Examples

Run the example scripts to see the framework in action:

```bash
# Train on circles dataset
cd examples
python train_circles.py

# IMDB sentiment analysis (requires movie_review_data.json)
python imdb_sentiment.py
```

## Architecture

### Forward Pass

The network performs a forward pass by sequentially applying each layer's transformation:

```
Input → Dense → ReLU → Dense → ReLU → Dense → Sigmoid → Output
```

### Backward Pass

Gradients are computed using backpropagation:

1. Compute loss gradient from loss function
2. Propagate gradients backward through layers in reverse order
3. Each layer computes:
   - Gradient w.r.t. inputs (passed to previous layer)
   - Gradient w.r.t. parameters (used for updates)
4. Optimizer applies gradients to update parameters

### Optimizers

**SGD (Stochastic Gradient Descent)**
```
θ = θ - learning_rate * gradient
```

**Adam (Adaptive Moment Estimation)**
```
m = β₁ * m + (1 - β₁) * g
v = β₂ * v + (1 - β₂) * g²
θ = θ - α * m̂ / (√v̂ + ε)
```

## Implementation Details

### Parameter Class

The `Parameter` class extends NumPy arrays with gradient tracking:

```python
w = Parameter(np.ones(3), name="weights")
w.gradient  # Access gradients
w.zero_gradient()  # Reset gradients
w.apply_gradient(grad)  # Apply gradient update
```

### Layer Interface

All layers implement:
- `forward(x)`: Compute output from input
- `backward(grad_output)`: Compute gradients

### Network Training

The `Network` class coordinates:
- Forward pass through all layers
- Loss computation
- Backward pass (backpropagation)
- Parameter updates via optimizer

## Testing

Run unit tests (if implemented):

```bash
pytest tests/
```

## Contributing

This is an educational project. Feel free to:
- Add new layer types (Conv2D, LSTM, etc.)
- Implement additional optimizers (RMSprop, AdaGrad, etc.)
- Add more loss functions (CrossEntropy, etc.)
- Improve documentation and examples

## License

MIT License - feel free to use for educational purposes.

## Acknowledgments

- Based on homework assignment from CSCI 4622 (Machine Learning) - Fall 2025
- Inspired by PyTorch's module design
- Uses scikit-learn conventions for feature transformers

## References

- [Understanding Backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Adam Optimizer Paper](https://arxiv.org/pdf/1412.6980.pdf)
- [Scikit-learn Custom Transformers](https://scikit-learn.org/stable/developers/develop.html)
