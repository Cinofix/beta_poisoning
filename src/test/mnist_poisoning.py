# !/usr/bin/env python
# coding: utf-8

import datetime

from secml.ml.classifiers import CClassifierSVM
from secml.adv.attacks import CAttackPoisoningSVM
from src.classifier.secml_classifier import LogisticClassifier
from secml.ml.peval.metrics import CMetricAccuracy
from data.cifar_loader import *
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from src.classifier.clf_utilities import test_clf, test_poison
from data.mnist_loader import *


def train_clf(tr):
    clf_ = CClassifierSVM(C=100, kernel="linear")  # CClassifierLogistic(C=100)  #
    clf = LogisticClassifier(clf_)
    print("Training of classifier...")
    clf.fit(tr)
    return clf


from src.experiments.utilities import set_seed

set_seed(444)

digits = (4, 0)

tr, val, ts = load_mnist(digits=digits, n_tr=400, n_val=1000, n_ts=1800)

print(tr.X.shape, val.X.shape)
# In[train clf]
clf = train_clf(tr)

# In[experiments clf]
acc = test_clf(clf, ts)
print("Accuracy on test set: {:.2%}".format(acc))
y_pred = clf.predict(ts.X)

# Should be chosen depending on the optimization problem
lb, ub = (
    val.X.min(),
    val.X.max(),
)

n_poisoning_points = int(0.1 * tr.X.shape[0])  # Number of poisoning points to generate

solver_params = {
    "eta": 0.25,
    "eta_min": 2.0,
    "eta_max": None,
    "max_iter": 100,
    "eps": 1e-6,
}

# In[white poisoning]
clf_white = clf.clf  # ._binary_classifiers[0]
pois_attack = CAttackPoisoningSVM(
    classifier=clf_white,
    training_data=tr,
    val=val,
    lb=lb,
    ub=ub,
    solver_params=solver_params,
)
pois_attack.n_points = n_poisoning_points

# Run the poisoning attack
start = datetime.datetime.now()
print("Start whitebox: ", start)
pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(ts.X, ts.Y)
print(
    "END whitebox: ", datetime.datetime.now(), " --> ", datetime.datetime.now() - start
)

metric = CMetricAccuracy()
# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=ts.Y, y_pred=pois_y_pred)

print("\n")
print(
    "[White-box] Accuracy after attack on experiments set: {:.2%}".format(pois_acc),
    "\t time ",
    datetime.datetime.now() - start,
)
# In[show boundary]
# Training of the poisoned classifier
pois_clf = clf.deepcopy()
pois_tr = tr.append(pois_ds)  # Join the training set with the poisoning points
pois_clf.fit(pois_tr)

box = [0, 1]

n, m = val.X.shape
for i in range(3):
    start = datetime.datetime.now()
    x_poison, y_poison = beta_poison(
        tr,
        val,
        clf,
        box,
        n_poisoning_points,
        k=15,
        y_target=None,
        y_poison=None,
        verbose=False,
        transform=to_scaled_img,
    )
    time_required = datetime.datetime.now() - start

    clf_p, test_acc, val_acc = test_poison(clf, tr, val, ts, x_poison, y_poison)
    print(
        "[black-box] Accuracy after attack on test set: {:.2%}".format(test_acc),
        "\t time ",
        time_required,
    )
    print("[black-box] Accuracy after attack on val set: {:.2%}".format(val_acc))

# In[show poison]
import matplotlib.pyplot as plt

n_to_show = 5
n_rows = 3
fig, axs = plt.subplots(n_rows, n_to_show)

for r in range(n_rows):
    for j in range(n_to_show):
        i = r * n_rows + j
        axs[r, j].imshow(pois_ds.X[i, :].tondarray().reshape(28, 28), cmap="gray")
        label = clf_p.predict(CArray(x_poison[i]))  # int(y_poison[i])
        axs[r, j].set_title("Y$_p$ {}".format(digits[label.item()]))
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
f = axs[0].imshow(x_poison[3].reshape(28, 28), cmap="gray")
f.axes.get_xaxis().set_visible(False)
f.axes.get_yaxis().set_visible(False)

f = axs[1].imshow(pois_ds.X[1, :].tondarray().reshape(28, 28), cmap="gray")
f.axes.get_xaxis().set_visible(False)
f.axes.get_yaxis().set_visible(False)

plt.show()
