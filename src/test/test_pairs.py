import datetime
from src.classifier.clf_utilities import test_clf, test_poison
from secml.ml.classifiers import CClassifierSVM
from src.classifier.secml_classifier import SVMClassifier
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from data.cifar_loader import *
from data.mnist_loader import *
from src.experiments.utilities import set_seed


def train_clf(tr):
    clf_ = CClassifierSVM(C=100, kernel="linear")
    clf = SVMClassifier(clf_)
    print("Training of classifier...")
    clf.fit(tr)
    return clf


set_seed(444)

digits = (4, 0)
classes = (6, 8)
ds = "cifar"

if ds == "cifar":
    tr, val, ts = load_data(labels=classes, n_tr=300, n_val=500, n_ts=1500)
else:
    tr, val, ts = load_mnist(digits=digits, n_tr=400, n_val=1000, n_ts=1000)


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

# Number of poisoning points to generate
n_poisoning_points = int(0.1388 * tr.X.shape[0])

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
from secml.array import CArray

n_to_show = 2
n_rows = 2
fig, axs = plt.subplots(n_rows, n_to_show)

for r in range(n_rows):
    for j in range(n_to_show):
        i = r * n_rows + j
        if ds == "mnist":
            axs[r, j].imshow(x_poison[i].reshape(28, 28), cmap="gray")
        else:
            axs[r, j].imshow(x_poison[i].reshape(3, 32, 32).transpose(1, 2, 0))
        label = clf_p.predict(CArray(x_poison[i]))  # int(y_poison[i])
        axs[r, j].set_title("Y$_p$ {}".format(digits[label.item()]))
plt.show()
