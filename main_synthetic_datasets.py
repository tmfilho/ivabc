from __future__ import division
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd

from util.functions import generate_multivariate_gaussians
from models.ivabc import IVABC


parameters = {
    "dataset1": {
        "n_prots": [20, 20],
        "k": 3,
        "alpha": 0.6,
        "n_particles": 25,
        "max_iter": 500,
        "max_reps": 500,
        "gaussians": np.array([
            [99, 9, 99, 169, 200, 0],
            [108, 9, 99, 169, 200, 1]
        ])
    },
    "dataset2": {
        "n_prots": [20, 20],
        "k": 21,
        "alpha": 0.8,
        "n_particles": 25,
        "max_iter": 500,
        "max_reps": 500,
        "gaussians": np.array([
            [99, 9, 99, 169, 200, 0],
            [104, 16, 138, 16, 150, 0],
            [104, 16, 60, 16, 150, 1],
            [108, 9, 99, 169, 200, 1]
        ])
    },
    "dataset3": {
        "n_prots": [20, 20],
        "k": 2,
        "alpha": 0.2,
        "n_particles": 25,
        "max_iter": 500,
        "max_reps": 500,
        "gaussians": np.array([
            [99, 9, 99, 169, 44, 25, 200, 0],
            [104, 16, 138, 16, 44, 25, 150, 0],
            [104, 16, 60, 16, 41, 25, 150, 1],
            [108, 9, 99, 169, 41, 25, 200, 1]
        ])
    },
    "dataset4": {
        "n_prots": [20, 20],
        "k": 21,
        "alpha": 0.5,
        "n_particles": 25,
        "max_iter": 500,
        "max_reps": 500,
        "gaussians": np.array([
            [99, 9, 99, 169, 200, 0],
            [104, 16, 118, 16, 150, 0],
            [104, 16, 80, 16, 150, 1],
            [100, 9, 99, 169, 200, 1]
        ])
    }
}


mc_iterations = 1
n_folds = 10


if __name__ == '__main__':
    np.random.seed(1)
    dataset_name = "dataset4"
    print dataset_name
    params = parameters[dataset_name]
    dataset = generate_multivariate_gaussians(params["gaussians"], 10)
    X, y = dataset.data, dataset.target
    mins = dataset.data[:, ::2].min(0)
    maxs = dataset.data[:,1::2].max(0)
    skf = StratifiedShuffleSplit(test_size=0.5)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i == 0:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            ivabc = IVABC(params["n_particles"], params["n_prots"], mins,
                          maxs, params["alpha"], params["k"],
                          params["max_iter"], params["max_reps"])
            ivabc.fit(X_train, y_train)
            curve = ivabc.convergence_curve
            curve_data = np.hstack((curve.reshape(-1,1),
                                    np.arange(
                                        params["max_iter"]).reshape(
                                        -1,1)))
            df = pd.DataFrame(data=curve_data, columns=["fitness",
                                                        "iteration"])
            df.to_csv("convergence-curve-{}.csv".format(dataset_name),
                      header=None)
