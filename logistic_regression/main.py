import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    # Load dataset
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train model
    regressor = LogisticRegression(learning_rate=0.0001, n_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Evaluate
    print("Logistic Regression classification accuracy:", accuracy(y_test, predictions))

if __name__ == "__main__":
    main()
