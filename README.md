# Coding up a Neural Network classifier from scratch

<p align="center">
<img src="https://github.com/ankonzoid/NN-from-scratch/blob/master/images/NN.png" width="50%">
</p>
 
We build and train a single hidden fully-connected layer neural network (written from scratch ), to classify the seeds dataset (https://archive.ics.uci.edu/ml/datasets/seeds). An L2 loss function, sigmoid activation, and no bias terms are assumed. The weight optimization is gradient descent via the Delta Rule (gradient descent).

### Usage:

Run the command

> python3 NN_classifier.py

The output should look similar to:

```
Reading 'data/seeds_dataset.csv'...
 -> X.shape = (210, 7), y.shape = (210,), n_classes = 3

Neural network model:
 input_dim = 7
 hidden_layers = [5]
 output_dim = 3
 eta = 0.6
 n_epochs = 800
 n_folds = 4

Cross-validating...
 Fold 1/4: acc_train = 98.73%, acc_valid = 98.08% (n_train = 158, n_test = 52)
 Fold 2/4: acc_train = 98.73%, acc_valid = 98.08% (n_train = 158, n_test = 52)
 Fold 3/4: acc_train = 98.73%, acc_valid = 96.15% (n_train = 158, n_test = 52)
 Fold 4/4: acc_train = 99.37%, acc_valid = 94.23% (n_train = 158, n_test = 52)

Avg acc_train = 98.89%
Avg acc_valid = 96.63%
```

### Libraries required:

* numpy, pandas