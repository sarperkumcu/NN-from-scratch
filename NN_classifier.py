"""

 NN_classifier.py  (author: Anson Wong / git: ankonzoid)

 We train a multi-layer fully-connected neural network from scratch to classify
 the seeds dataset (https://archive.ics.uci.edu/ml/datasets/seeds). An L2 loss
 function, sigmoid activation, and no bias terms are assumed. The weight
 optimization is gradient descent via the delta rule.

"""
import numpy as np
from src.NeuralNetwork import NeuralNetwork
import src.utils as utils


def main():
    # ===================================
    # Settings
    # ===================================
    csv_filename = "data/creditcard.csv"
    hidden_layers = [5]
    eta = 0.1
    n_epochs = 500
    n_folds = 3

    X, y, n_classes = utils.read_csv(csv_filename, target_name="Class")
    N, d = X.shape
    print(" -> X.shape = {}, y.shape = {}, n_classes = {}\n".format(X.shape, y.shape, n_classes))

    print("Running")
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=1)

    acc_train, acc_valid = list(), list()
    print("Cross-validation")
    for i, idx_valid in enumerate(idx_folds):
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        model = NeuralNetwork(input_dim=d, output_dim=n_classes,
                              hidden_layers=hidden_layers, seed=1)
        model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)

        ypred_train = model.predict(X_train)
        ypred_valid = model.predict(X_valid)

        acc_train.append(100 * np.sum(y_train == ypred_train) / len(y_train))
        acc_valid.append(100 * np.sum(y_valid == ypred_valid) / len(y_valid))
        print("TP: " + str(np.sum((y_valid == ypred_valid) & (y_valid == 1))))
        print("TN: " + str(np.sum((y_valid == ypred_valid) & (y_valid == 0))))
        print("FP: " + str(np.sum((y_valid != ypred_valid) & (y_valid == 1))))
        print("FN: " + str(np.sum((y_valid != ypred_valid) & (y_valid == 0))))
        TP = np.sum((y_valid == ypred_valid) & (y_valid == 1))
        TN = np.sum((y_valid == ypred_valid) & (y_valid == 0))
        FP = np.sum((y_valid != ypred_valid) & (y_valid == 1))
        FN = np.sum((y_valid != ypred_valid) & (y_valid == 0))
        precision = calculate_precision(TP, FP)
        recall = calculate_recall(TP, FN)

        print(str(f1_score(recall, precision)))
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i + 1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

    print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(
        sum(acc_train) / float(len(acc_train)), sum(acc_valid) / float(len(acc_valid))))


def calculate_precision(TP, FP):
    return TP / (TP + FP)


def calculate_recall(TP, FN):
    return TP / (TP + FN)


def f1_score(recall, precision):
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main()