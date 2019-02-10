"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from csv import reader

#
# Read csv file as X, y data (last column is class label)
#
def read_csv(filename, normalize=False):
    X_str = list()  # data (float)
    y_str = list()  # class labels (integers)

    # Read X and y data from csv file
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for i, row in enumerate(csv_reader):
            if i == 0: # skip row if empty
                continue
            else:
                X_str.append(row[:-1])
                y_str.append(row[-1])

    # Convert our class labels to 0, 1, 2, ..., n_classes-1
    def convert_str2idx(y_str):
        unique = set(y_str)
        lookup = dict()
        # Assign each unique class label an index 0, 1, 2, ..., n_classes-1
        for idx_label, label in enumerate(unique):
            lookup[label] = idx_label
        y_idx = list()
        for label in y_str:
            y_idx.append(lookup[label])
        return y_idx

    y_idx = convert_str2idx(y_str)

    # Convert to numpy arrays
    X = np.array(X_str, dtype=np.float32)
    y = np.array(y_idx, dtype=np.int)
    n_classes = len(np.unique(y))

    # Normalize (optional)
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y, n_classes

#
# Randomly permute and extract indices for each fold
#
def crossval_folds(N, n_folds, seed=1):
    np.random.seed(seed)
    idx_all_permute = np.random.permutation(N)
    N_fold = int(N/n_folds)
    idx_folds = []
    for i in range(n_folds):
        start = i*N_fold
        end = min([(i+1)*N_fold, N])
        idx_folds.append(idx_all_permute[start:end])
    return idx_folds