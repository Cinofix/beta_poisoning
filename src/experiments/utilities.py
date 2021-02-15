import random
import time
import numpy as np
import torch
from secml.ml.peval.metrics import CMetricConfusionMatrix


def list_to_string(lst):
    return '"[' + ",".join("%.4f" % x for x in lst) + ']"'


def timeit(f, x):
    start = time.time()
    out = f(x)
    exec_time = time.time() - start
    return out, exec_time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def length(ds):
    return ds.X.shape[0]


def accuracy_by_labels(y_true, y_pred):
    cfm = CMetricConfusionMatrix()
    scores = cfm.performance_score(y_true, y_pred)
    return scores.diag() / scores.sum(axis=1).flatten()


import sys
from optparse import OptionParser

# parse commandline arguments
op = OptionParser()
op.add_option(
    "--device",
    type=str,
    default="cpu",
    help="Set device name on which experiments are performed.",
)
op.add_option(
    "--path",
    type=str,
    default="./IJCNN_Experiments",
    help="Destination path, where results will be stored.",
)
op.add_option(
    "--generator",
    type=str,
    default="beta",
    help="Poisoning generator. "
    "- [white] for white-box attack"
    "- [beta] for proto-poisoning attack"
    "- [flip] for label flip attack",
)
op.add_option(
    "--classifier", type=str, default="svm", help="Attack 'svm' or 'logistic'",
)
op.add_option(
    "--lb",
    type=float,
    default=0.01,
    help="Regularization parameter for closeness to kde.",
)
op.add_option(
    "--n_proto", type=int, default=10, help="Number of prototypes.",
)

op.add_option(
    "--ds",
    type=str,
    default="4-0",
    help="Dataset subset. Classes are separated by `-`. Ex: 4-0 refers to classes 4 and 0",
)

(opts, args) = op.parse_args(sys.argv[1:])
