"""

 NN_classifier.py  (author: Anson Wong / git: ankonzoid)

 Building and training a neural network classifier with back-propagation
 from scratch. The loss function is assumed to be L2-norm, and we
 do not include any biases in the activation calculation. Also,
 a sigmoid transfer function is used on all nodes. The delta rule (gradient
 descent) is used as our weight update rule is gradient descent on
 L2-loss function. More on the delta rule can
 be found at: https://en.wikipedia.org/wiki/Delta_rule.

 The format of the input data should be floats except for the last column
 which should be an integer to represent the class labels. The pre-processing
 of the data also normalizes each feature column of X.

 This code was inspired by:
 https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from NeuralNetworkClass import NeuralNetwork
import utils

def main():
    # ===================================
    # Settings
    # ===================================
    filename = "data/seeds_dataset.csv"
    l_rate = 0.6  # learning rate
    n_epoch = 800  # training epochs
    n_hidden = 5  # nodes in hidden layer
    n_hidden_layers = 1  # number of hidden layers
    n_folds = 4  # number of folds for cross-validation

    # ===================================
    # Read data (X,y) and normalize X
    # ===================================
    print("Reading '{}'...".format(filename))
    X, y = utils.read_csv(filename)  # read as matrix of floats and int
    utils.normalize(X)  # normalize
    (N, d) = X.shape  # extract shape of X
    n_classes = len(np.unique(y))

    print(" X.shape = {}".format(X.shape))
    print(" y.shape = {}".format(y.shape))
    print(" n_classes = {}".format(n_classes))

    # ===================================
    # Create cross-validation folds
    # These are a list of a list of indices for each fold
    # ===================================
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=1)

    # ===================================
    # Train and evaluate the model on each fold
    # ===================================
    accuracy_train = list()
    accuracy_test = list()
    print("\nTraining and cross-validating...")
    for i, idx_test in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_test)
        (X_train, y_train) = (X[idx_train], y[idx_train])
        (X_test, y_test) = (X[idx_test], y[idx_test])

        # Set architecture and train NN model
        model = NeuralNetwork(n_input=d,
                              n_output=n_classes,
                              n_hidden=n_hidden,
                              n_hidden_layers=n_hidden_layers)
        model.build_network()
        model.train(X_train, y_train, l_rate=l_rate, n_epoch=n_epoch)

        # Make predictions for training and test data
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        # Compute test accuracy score from predicted values
        accuracy_train.append(100*np.sum(y_train==y_train_predict)/len(y_train))
        accuracy_test.append(100*np.sum(y_test==y_test_predict)/len(y_test))

        # Print cross-validation result
        print(" Fold {}/{}: train acc = {:.2f}%, test acc = {:.2f}% (n_train = {}, n_test = {})".format(i+1, n_folds, accuracy_train[-1], accuracy_test[-1], len(X_train), len(X_test)))

    # ===================================
    # Print results
    # ===================================
    print("\nAvg train acc = {:.2f}%".format(sum(accuracy_train) / float(len(accuracy_train))))
    print("Avg test acc = {:.2f}%".format(sum(accuracy_test) / float(len(accuracy_test))))


# Driver
if __name__ == "__main__":
    main()