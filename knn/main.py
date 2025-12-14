#!/usr/bin/env python
"""
Main demonstration script for KNN implementations.

This script demonstrates the usage of KNN classifiers and regressors
on various datasets including binary classification, digit recognition,
and housing price prediction.

Usage:
    python main.py [--task TASK] [--k K]

    TASK: 'binary', 'digits', 'housing', or 'all' (default: 'all')
    K: Number of neighbors (default: auto-select based on validation)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from knn import KNNClassifier, WeightedKNNClassifier, KNNRegressor
from knn import evaluate, prepare_data, plot_k_vs_metric
import data
import helpers


def demo_binary_classification(k=None):
    """
    Demonstrate KNN on binary classification task.

    Parameters
    ----------
    k : int, optional
        Number of neighbors. If None, will be auto-selected.
    """
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION DEMO")
    print("="*70)

    # Load data
    binary_data = data.BinaryData()
    print(f"Training samples: {len(binary_data.X_train)}")
    print(f"Validation samples: {len(binary_data.X_valid)}")
    print(f"Test samples: {len(binary_data.X_test)}")

    if k is None:
        # Model selection
        print("\nPerforming model selection...")
        ks = list(range(1, 32))
        accuracies_train = []
        accuracies_valid = []

        for k_val in ks:
            knn = KNNClassifier(k_val).fit(binary_data.X_train, binary_data.y_train)
            acc_train = knn.accuracy(binary_data.X_train, binary_data.y_train)
            acc_valid = knn.accuracy(binary_data.X_valid, binary_data.y_valid)
            accuracies_train.append(acc_train)
            accuracies_valid.append(acc_valid)

        best_k = ks[np.argmax(accuracies_valid)]
        print(f"Best k from validation: {best_k}")

        # Plot k vs accuracy
        plot_k_vs_metric(ks, accuracies_train, accuracies_valid,
                        metric_name="Accuracy",
                        title="Binary Classification: Accuracy vs k")
        plt.savefig("binary_k_vs_accuracy.png")
        print("Saved plot: binary_k_vs_accuracy.png")
    else:
        best_k = k

    # Train final model
    knn = KNNClassifier(best_k).fit(binary_data.X_train, binary_data.y_train)
    acc_test = knn.accuracy(binary_data.X_test, binary_data.y_test)
    print(f"\nTest Accuracy (k={best_k}): {acc_test:.4f}")

    # Show confusion matrix
    cm = knn.confusion_matrix(binary_data.X_test, binary_data.y_test)
    print(f"Confusion Matrix:\n{cm}")

    # Visualize decision surface
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    helpers.show_decision_surface(knn, binary_data.X_train, binary_data.y_train, axs[0])
    axs[0].set_title("Training Set")
    helpers.show_decision_surface(knn, binary_data.X_valid, binary_data.y_valid, axs[1])
    axs[1].set_title("Validation Set")
    helpers.show_decision_surface(knn, binary_data.X_test, binary_data.y_test, axs[2])
    axs[2].set_title("Test Set")
    plt.tight_layout()
    plt.savefig("binary_decision_surface.png")
    print("Saved plot: binary_decision_surface.png")


def demo_digit_classification(k=None):
    """
    Demonstrate KNN on digit classification task.

    Parameters
    ----------
    k : int, optional
        Number of neighbors. If None, will be auto-selected.
    """
    print("\n" + "="*70)
    print("DIGIT CLASSIFICATION DEMO")
    print("="*70)

    # Load and prepare data
    digit_data = data.DigitData()
    digit_data = prepare_data(digit_data)

    if k is None:
        print("\nPerforming model selection...")
        evaluate(range(2, 10), digit_data, KNNClassifier)
    else:
        knn = KNNClassifier(k).fit(digit_data.X_train, digit_data.y_train)
        acc_test = knn.accuracy(digit_data.X_test, digit_data.y_test)
        print(f"\nTest Accuracy (k={k}): {acc_test:.4f}")

        cm = knn.confusion_matrix(digit_data.X_test, digit_data.y_test)
        print("Confusion Matrix:")
        helpers.display_confusion(cm)


def demo_weighted_knn():
    """
    Compare standard KNN with WeightedKNN on binary data.
    """
    print("\n" + "="*70)
    print("WEIGHTED KNN COMPARISON")
    print("="*70)

    binary_data = data.BinaryData()

    print("\nStandard KNN Classifier:")
    evaluate(range(2, 10), binary_data, KNNClassifier)

    print("\nWeighted KNN Classifier:")
    evaluate(range(2, 10), binary_data, WeightedKNNClassifier)


def demo_housing_regression(k=None):
    """
    Demonstrate KNN Regressor on housing price prediction.

    Parameters
    ----------
    k : int, optional
        Number of neighbors. If None, will be auto-selected.
    """
    print("\n" + "="*70)
    print("HOUSING PRICE REGRESSION DEMO")
    print("="*70)

    # Load data
    housing_data = data.HousingPrices()
    print(f"Training samples: {len(housing_data.X_train)}")
    print(f"Validation samples: {len(housing_data.X_valid)}")
    print(f"Test samples: {len(housing_data.X_test)}")
    print(f"Number of features: {housing_data.X_train.shape[1]}")

    if k is None:
        # Model selection
        print("\nPerforming model selection...")
        ks = list(range(1, 32))
        mse_train = []
        mse_valid = []

        for k_val in ks:
            print(f"k={k_val}", end="\r")
            knn_reg = KNNRegressor(k_val).fit(housing_data.X_train, housing_data.y_train)
            mse_train.append(knn_reg.mse(housing_data.X_train, housing_data.y_train))
            mse_valid.append(knn_reg.mse(housing_data.X_valid, housing_data.y_valid))

        best_k = ks[np.argmin(mse_valid)]
        print(f"Best k from validation: {best_k}")

        # Plot k vs MSE
        plot_k_vs_metric(ks, mse_train, mse_valid,
                        metric_name="MSE",
                        ylabel="Mean Squared Error",
                        title="Housing Regression: MSE vs k")
        plt.savefig("housing_k_vs_mse.png")
        print("Saved plot: housing_k_vs_mse.png")
    else:
        best_k = k

    # Train final model
    knn_reg = KNNRegressor(best_k).fit(housing_data.X_train, housing_data.y_train)
    mse_test = knn_reg.mse(housing_data.X_test, housing_data.y_test)
    rmse_test = np.sqrt(mse_test)

    print(f"\nTest MSE (k={best_k}): {mse_test:.4f}")
    print(f"Test RMSE (k={best_k}): {rmse_test:.4f}")
    print(f"(In units of $100,000)")


def main():
    """Main entry point for the demonstration script."""
    parser = argparse.ArgumentParser(
        description='Demonstrate KNN implementations on various datasets'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['binary', 'digits', 'housing', 'weighted', 'all'],
        help='Which task to run (default: all)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Number of neighbors (default: auto-select)'
    )

    args = parser.parse_args()

    print("="*70)
    print("KNN DEMONSTRATIONS")
    print("="*70)

    if args.task in ['binary', 'all']:
        demo_binary_classification(k=args.k)

    if args.task in ['digits', 'all']:
        demo_digit_classification(k=args.k)

    if args.task in ['weighted', 'all']:
        demo_weighted_knn()

    if args.task in ['housing', 'all']:
        demo_housing_regression(k=args.k)

    print("\n" + "="*70)
    print("DEMONSTRATIONS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
