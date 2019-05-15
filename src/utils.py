import numpy as np
import pandas as pd

# Read csv file for (X, y, n_classes) data
def read_csv(csv_filename, target_name="y"):

    # Read csv
    old_df = pd.read_csv(csv_filename, delimiter=",", dtype={target_name: np.str})
    print(old_df)
    df_undersampled = old_df[old_df['Class'] == "0"]
    a = df_undersampled[:2000]
    df_fraud = old_df[old_df['Class'] == "1"]
    df = pd.concat([df_fraud, df_fraud])
    df = pd.concat([a, df])
    df = df.drop(['Time','Amount'], axis=1)
    print(df)



    # Check target exists
    if list(df.columns.values).count(target_name) != 1:
        raise Exception("Need exactly 1 count of '{}' in {}".format(target_name, csv_filename))

    # Create (target -> index) mapping
    map = {}
    targets_unique = sorted(list(set(df[target_name].values)))
    for i, target in enumerate(targets_unique):
        map[target] = i

    def class2idx(y_, map):
        if y_ in map.keys(): return map[y_]
        else: raise Exception("Invalid key provided!")

    X = df.drop([target_name], axis=1).values
    y = np.vectorize(class2idx)(df[target_name], map)
    n_classes = len(map.keys())

    if X.shape[0] != y.shape[0]:
        raise Exception("X.shape = {} and y.shape = {} are inconsistent!".format(X.shape, y.shape))

    return X, y, n_classes

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