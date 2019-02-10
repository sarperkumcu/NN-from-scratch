"""

 NN_classifier.py  (author: Anson Wong / git: ankonzoid)

 Building and training a neural network classifier using back-propagation
 from scratch. An L2 loss, no bias terms, and sigmoid activations are used.
 Optimization of this neural network is via gradient descent (delta rule).

 A csv reader is included that takes rows of floats except for the last column
 which should be an integer to represent the class labels.
 The input features are normalized before feeding into the neural network.

"""
import numpy as np
import pandas as pd
from src.NeuralNetwork import NeuralNetwork
import src.utils as utils

def main():
    # ===================================
    # Settings
    # ===================================
    csv_filename = "data/seeds_dataset.csv"
    hidden_layers = [5] # nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = 0.6 # learning rate
    n_epochs = 800 # number of training epochs
    n_folds = 4 # number of folds for cross-validation

    # ===================================
    # Read csv data + normalize features
    # ===================================
    print("Reading '{}'...".format(csv_filename))
    X, y, n_classes = utils.read_csv(csv_filename, normalize=True)
    N, d = X.shape
    print(" -> X.shape = {}, y.shape = {}, n_classes = {}\n".format(X.shape, y.shape, n_classes))

    print("Neural network model:")
    print(" input_dim = {}".format(d))
    print(" hidden_layers = {}".format(hidden_layers))
    print(" output_dim = {}".format(n_classes))
    print(" eta = {}".format(eta))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}\n".format(n_folds))

    # ===================================
    # Create cross-validation folds
    # These are a list of a list of indices for each fold
    # ===================================
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=1)

    # ===================================
    # Train and evaluate the model on each fold
    # ===================================
    acc_train, acc_valid = list(), list()  # training/test accuracy score
    print("Cross-validating...")
    for i, idx_valid in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Build neural network classifier model and train
        model = NeuralNetwork(input_dim=d, output_dim=n_classes, hidden_layers=hidden_layers)
        model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)

        # Make predictions for training and test data
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_valid)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==y_train_predict)/len(y_train))
        acc_valid.append(100*np.sum(y_valid==y_test_predict)/len(y_valid))

        # Print cross-validation result
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_test = {})".format(i+1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

    # ===================================
    # Print results
    # ===================================
    print("\nAvg acc_train = {:.2f}%".format(sum(acc_train)/float(len(acc_train))))
    print("Avg acc_valid = {:.2f}%".format(sum(acc_valid)/float(len(acc_valid))))

# Driver
if __name__ == "__main__":
    main()