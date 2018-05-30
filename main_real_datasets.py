from __future__ import division
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd

from util.functions import get_dataset
from models.ivabc import IVABC
from util.functions import print_confusion_matrix


parameters = {
    "mushroom": {
        "n_prots": [7, 2],
        "k": 3,
        "alpha": 0.0,
        "n_particles": 25,
        "max_iter": 200,
        "max_reps": 50
    },
    "european-climates": {
        "n_prots": [20, 40],
        "k": 3,
        "alpha": 0.2,
        "n_particles": 25,
        "max_iter": 200,
        "max_reps": 50
    },
    "dry-climates": {
        "n_prots": [28, 38, 34],
        "k": 21,
        "alpha": 0.5,
        "n_particles": 25,
        "max_iter": 200,
        "max_reps": 50
    },
    "all-climates": {
        "n_prots": [34, 28, 12, 20, 12, 42, 38, 34, 6, 40],
        "k": 8,
        "alpha": 1.0,
        "n_particles": 25,
        "max_iter": 200,
        "max_reps": 50
    }
}


mc_iterations = 1
n_folds = 10


if __name__ == '__main__':
    np.random.seed(1)
    dataset_name = "all-climates"
    print dataset_name
    params = parameters[dataset_name]
    filename = "dados/" + dataset_name + "-headers.csv"
    dataset = get_dataset(filename)
    X, y = dataset.data, dataset.target
    mins = dataset.data[:, ::2].min(0)
    maxs = dataset.data[:,1::2].max(0)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i == 0:
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            preds = np.zeros((30, len(y_test)))
            from tqdm import tqdm
            for l in tqdm(np.arange(30)):
                ivabc = IVABC(params["n_particles"], params["n_prots"], mins,
                              maxs, params["alpha"], params["k"],
                              params["max_iter"], params["max_reps"])
                ivabc.fit(X_train, y_train)
                predictions = ivabc.predict(X_test)
                preds[l] = predictions
            predictions = np.around(np.mean(preds, axis=0))
            print_confusion_matrix(y_test, predictions)
            # curve = ivabc.convergence_curve
            # curve_data = np.hstack((curve.reshape(-1, 1),
            #                         np.arange(
            #                             params["max_iter"]).reshape(
            #                             -1,1)))
            # df = pd.DataFrame(data=curve_data, columns=["fitness",
            #                                             "iteration"])
            # df.to_csv("convergence-curve-{}.csv".format(dataset_name),
            #           header=None)
