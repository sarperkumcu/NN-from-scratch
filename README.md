# Coding from scratch a single hidden-layer neural network classifier
 
We build and train a single hidden fullly-connected layer neural network classifier written from scratch (no high-level libraries such as tensorflow, keras, pytorch, etc), and train it via back-propagation on the "seeds_dataset.csv" (https://archive.ics.uci.edu/ml/datasets/seeds) data set containing measurements of geometrical properties of kernels belonging to three different varieties of wheat. The loss function is assumed to be L2-norm, and we do not include any biases in the activation calculation. Also, a sigmoid transfer function is used on all nodes.

 If you would like to import your own dataset, then follow the format of the data being entirely of floats except for the last column which should be only integer class labels. The code does pre-processing of the data X by normalizing each feature column.

 This code was inspired by another implementation:
 https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

### Usage:

Run the command

> python NN_classifier.py

The output should look similar to:

```
Reading 'data/seeds_dataset.csv'...
X.shape = (210, 7)
y.shape = (210,)
Training and cross-validating...
Fold 1/4: train acc = 87.34%, test acc = 86.54% (n_train = 158, n_test = 52)
Fold 2/4: train acc = 91.77%, test acc = 96.15% (n_train = 158, n_test = 52)
Fold 3/4: train acc = 88.61%, test acc = 78.85% (n_train = 158, n_test = 52)
Fold 4/4: train acc = 88.61%, test acc = 75.00% (n_train = 158, n_test = 52)

Avg train acc = 89.08%
Avg test acc = 84.13%

```

### Libraries required:

* numpy
