"""
Unit tests for layer implementations.

This module contains tests for Dense, Sigmoid, and ReLU layers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mlp import Dense, Sigmoid, ReLU


def test_sigmoid_forward():
    """Test Sigmoid forward pass."""
    mock_X = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    expected = np.array([[0.38133797, 0.52019699], [0.71799983, 0.37679803]])

    sigmoid = Sigmoid()
    output = sigmoid.forward(mock_X)

    assert np.allclose(output, expected), "Sigmoid forward pass failed"
    print("✓ Sigmoid forward pass test passed")


def test_sigmoid_backward():
    """Test Sigmoid backward pass."""
    mock_X = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    grad_output = np.array([[0.19960269, 0.20993069], [-0.85814751, -0.41418101]])
    expected_grad = np.array([[0.04709013, 0.05239704], [-0.17375434, -0.09725851]])

    sigmoid = Sigmoid()
    sigmoid.forward(mock_X)
    grad_input = sigmoid.backward(grad_output)

    assert np.allclose(grad_input, expected_grad), "Sigmoid backward pass failed"
    print("✓ Sigmoid backward pass test passed")


def test_relu_forward():
    """Test ReLU forward pass."""
    mock_X = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    expected = np.array([[0., 0.08083195], [0.93456167, 0.]])

    relu = ReLU()
    output = relu.forward(mock_X)

    assert np.allclose(output, expected), "ReLU forward pass failed"
    print("✓ ReLU forward pass test passed")


def test_relu_backward():
    """Test ReLU backward pass."""
    mock_X = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    grad_output = np.array([[0.19960269, 0.20993069], [-0.85814751, -0.41418101]])
    expected_grad = np.array([[0., 0.20993069], [-0.85814751, 0.]])

    relu = ReLU()
    relu.forward(mock_X)
    grad_input = relu.backward(grad_output)

    assert np.allclose(grad_input, expected_grad), "ReLU backward pass failed"
    print("✓ ReLU backward pass test passed")


def test_dense_layer():
    """Test Dense layer forward and backward pass."""
    mock_X = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    grad_output = np.array([[0.19960269, 0.20993069], [-0.85814751, -0.41418101]])

    np.random.seed(42)
    dense = Dense(2, 2)

    # Forward pass
    output = dense.forward(mock_X)
    assert output.shape == (2, 2), "Dense layer output shape incorrect"

    # Backward pass
    grad_input = dense.backward(grad_output)
    assert grad_input.shape == mock_X.shape, "Dense layer gradient shape incorrect"
    assert dense.weights.gradient.shape == dense.weights.shape, "Weights gradient shape incorrect"
    assert dense.bias.gradient.shape == dense.bias.shape, "Bias gradient shape incorrect"

    print("✓ Dense layer test passed")


def run_all_tests():
    """Run all layer tests."""
    print("=" * 60)
    print("Running Layer Tests")
    print("=" * 60)

    test_sigmoid_forward()
    test_sigmoid_backward()
    test_relu_forward()
    test_relu_backward()
    test_dense_layer()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
