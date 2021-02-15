import sys

sys.path.extend(["./"])

from data.mnist_loader import *
from src.experiments.utilities import set_seed
from src.classifier.secml_classifier import SVMClassifier
from src.experiments.ablation.ablation_utilities import run_ablation
from src.optimizer.beta_optimizer import beta_poison


def mnist_bin(p_generator, clf, name, proto_lst=None):
    set_seed(444)
    tr_size = 400
    n_p = int(0.25 * tr_size)
    digits = (4, 0)
    tr_size, val_size, ts_size = [tr_size, 1000, None]
    ds = load_mnist(digits=digits, n_tr=tr_size, n_val=val_size, n_ts=ts_size)
    tr, val, ts = ds

    c = 1
    clf.init_fit(tr, parameters={"C": c})
    c_name = name + "_c" + str(c)
    run_ablation(
        p_generator, clf, c, ds, c_name, box=[0, 1], proto_lst=proto_lst, n_poison=n_p
    )


if __name__ == "__main__":
    kernel = "linear"
    clf = SVMClassifier(k=kernel)
    mnist_bin(
        beta_poison,
        clf,
        name="IJCNN_beta_mnist_ablation_%s" % kernel,
        proto_lst=[2, 5, 10, 15, 20, 25, 30],
    )
