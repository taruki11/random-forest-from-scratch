"""
Random Forest from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold for: Random Forest from Scratch.

Build a small synthetic dataset, train a random forest, predict, and report
train/test accuracy. Every function is concatenated above this scaffold, so
they are called directly (there is no separate solution module).
"""

import numpy as np


def main():
    rng = np.random.default_rng(0)
    n, d = 150, 5
    X = rng.random((n, d))
    y = ((X[:, 0] + X[:, 3]) > 1.0).astype(int)
    split = 110
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    forest = train_forest(X_train, y_train, num_trees=15, max_depth=6, random_state=0)
    print("forest size:", len(forest))

    train_acc = accuracy(predict_forest(forest, X_train), y_train)
    test_acc = accuracy(predict_forest(forest, X_test), y_test)
    print("train accuracy:", round(float(train_acc), 4))
    print("test accuracy: ", round(float(test_acc), 4))

    single = build_tree(X_train, y_train, max_depth=6)
    print("single-tree train accuracy:", round(float(accuracy(predict_tree(single, X_train), y_train)), 4))


if __name__ == "__main__":
    main()

