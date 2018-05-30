from __future__ import division

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


class Dataset:
    def __init__(self, dataframe=None, data=None, target=None):
        if dataframe is not None:
            data_columns = dataframe.columns[dataframe.columns != "label"]
            self.data = dataframe[data_columns].values
            self.target = dataframe["label"].values
        else:
            self.data = data
            self.target = target


def get_dataset(filename):
    data = pd.read_csv(filename,  header=0)
    return Dataset(dataframe=data)


def generate_multivariate_gaussians(parameters, interval):
    quantities = parameters[:, -2]
    n = np.sum(quantities)
    p = parameters[:, :-2].shape[1]
    labels = parameters[:, -1]
    data = np.zeros((n, p))
    target = np.zeros(n).astype(int)
    n_gaussians = parameters.shape[0]

    end = 0
    params = parameters[:, :-2]
    for gaussian in np.arange(n_gaussians):
        init = end
        end += quantities[gaussian]
        means = params[gaussian, ::2]
        covs = np.diag(params[gaussian, 1::2].astype(float))
        n_g = quantities[gaussian]
        d = np.random.multivariate_normal(means, covs, n_g)

        delta = np.random.randint(1, interval, int(p / 2))
        data[init:end, ::2] = d - (delta / 2.0)
        data[init:end, 1::2] = d + (delta / 2.0)
        target[init:end] = labels[gaussian]
    return Dataset(data=data, target=target)


def print_confusion_matrix(y_true, y_pred, classes=None, title="",
                           caption=""):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    if classes is None:
        classes = labels + 1
    print "\\begin{table}[H]"
    print "    \\centering"
    print "    \\caption{\\textbf{" + title + "} - " + caption + "}"
    s = "    \\begin{tabular}{c"
    for label in labels:
        s += "|c"
    print s + "|c}"
    s = "    \\toprule "
    for c in classes:
        s += "& {} ".format(c)
    print s + " & Total \\\\"
    print "    \\midrule"
    p = "Predicoes "
    for i, l1 in enumerate(labels):
        s = "{} ".format(classes[l1])
        for l2 in labels:
            s += "& {} ".format(cm[l1, l2])
        s += "& {} \\\\".format(np.sum(cm[l1]))
        print s + " \\midrule"
        p += "& {} ".format(np.sum(cm[:, l1]))
    p += "& {} \\\\".format(np.sum(cm))
    print p
    print "    \\bottomrule"
    print "    \\end{tabular}"
    print "    \\label{tab:tab}"
    print "\\end{table}"
