import sys

sys.path.extend(["./"])

from data.cifar_loader import *
from src.classifier.secml_classifier import SVMClassifier
from src.experiments.ablation.ablation_utilities import run_ablation
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.utilities import set_seed


def cifar_bin(p_generator, clf, name, proto_lst=None):
    set_seed(444)
    tr_size = 300
    n_p = int(0.25 * tr_size)

    digits = (6, 8)
    ds = load_data(labels=digits, n_tr=tr_size, n_val=1000, n_ts=1000)
    tr, val, ts = ds
    c = 1
    clf.init_fit(tr, parameters={"C": c})
    c_name = name + "_c"+str(c)
    run_ablation(
        p_generator, clf, c, ds, c_name, box=[0, 1], proto_lst=proto_lst, n_poison=n_p
    )


if __name__ == "__main__":
    kernel = "linear"
    clf = SVMClassifier()
    cifar_bin(
        beta_poison,
        clf,
        name="IJCNN_beta_cifar_ablation_%s" % kernel,
        proto_lst=[5, 10, 15, 20, 25, 30],
    )
