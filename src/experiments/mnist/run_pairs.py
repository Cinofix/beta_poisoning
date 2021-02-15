import sys

sys.path.extend(["./"])

from data.mnist_loader import *
from src.experiments.run_attack import *
from src.classifier.secml_classifier import SVMClassifier, LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from src.optimizer.flip_poisoning import flip_batch_poison
from src.optimizer.white_poisoning import white_poison
import os

if __name__ == "__main__":
    set_seed(444)
    d1, d2 = int(opts.ds[0]), int(opts.ds[2])
    digits = (d1, d2)
    tr, val, ts = load_mnist(digits=digits, n_tr=400, n_val=1000, n_ts=1000)

    clf = LogisticClassifier() if opts.classifier == "logistic" else SVMClassifier(k="linear")
    params = {
        "n_proto": opts.n_proto,
        "lb": 1,
        "y_target": None,
        "y_poison": None,
        "transform": to_scaled_img,
    }
    path = opts.path + "/mnist-{}-tr{}/{}/".format(
        opts.ds, tr.X.shape[0], opts.classifier
    )
    os.makedirs(path, exist_ok=True)

    if "beta" in opts.generator:
        name = path + "beta_poison_k" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, params=params)
    if "white" in opts.generator:
        name = path + "white_poison"
        run_attack(white_poison, name, clf, tr, val, ts, params=params)
    if "flip" in opts.generator:
        name = path + "flip"
        run_attack(flip_batch_poison, name, clf, tr, val, ts, params=params)
