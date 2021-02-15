from secml.ml.classifiers import CClassifierLogistic
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks import CAttackPoisoningLogisticRegression
from secml.figure import CFigure
from secml.array import CArray

random_state = 999

n_features = 2  # Number of features
n_samples = 1000  # Number of samples
centers = [[-1.5, -1.5], [+1.5, +1.5]]  # Centers of the clusters
cluster_std = 0.9  # Standard deviation of the clusters

from secml.data.loader import CDLRandomBlobs

dataset = CDLRandomBlobs(
    n_features=n_features,
    centers=centers,
    cluster_std=cluster_std,
    n_samples=n_samples,
    random_state=random_state,
).load()

n_tr = 300  # Number of training set samples
n_val = 500  # Number of validation set samples
n_ts = 200  # Number of test set samples

# Split in training, validation and test
from secml.data.splitter import CTrainTestSplit

splitter = CTrainTestSplit(
    train_size=n_tr + n_val, test_size=n_ts, random_state=random_state
)
tr_val, ts = splitter.split(dataset)
splitter = CTrainTestSplit(train_size=n_tr, test_size=n_val, random_state=random_state)
tr, val = splitter.split(dataset)

clf = CClassifierLogistic()
# clf = CClassifierSVM(kernel='rbf')
# clf.kernel.gamma = 0.1
clf.fit(tr.X, tr.Y)
print("Training of classifier complete!")

# Compute predictions on a test set
y_pred, scores = clf.predict(ts.X, return_decision_function=True)

clf_src = clf.deepcopy()

# Evaluate the accuracy of the classifier
metric = CMetricAccuracy()
acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
print("Accuracy on test set: {:.2%}".format(acc))

# Should be chosen depending on the optimization problem
solver_params = {"eta": 0.1, "max_iter": 1000, "eps": 1e-4}

# set box constraint on the attack point
lb, ub = -5, 5

# pois_attack = CAttackPoisoningSVM(
pois_attack = CAttackPoisoningLogisticRegression(
    classifier=clf,
    training_data=tr,
    val=val,
    lb=lb,
    ub=ub,
    solver_type="pgd",
    solver_params=solver_params,
    random_seed=random_state,
    init_type="random",
)

pois_attack.n_points = 1
# pois_attack.verbose = 1

xc = CArray([-1, -1])
yc = CArray(1)

from secml.data import CDataset

pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(
    x=val.X, y=val.Y, ds_init=CDataset(xc, yc)
)

# %%

width = 6
height = 5
# %%

from src.optimizer.kde import KDEGaussian
from src.classifier.secml_classifier import LogisticClassifier
from src.classifier.secml_autograd import as_tensor


class LikelihoodLoss:
    def __init__(self, val, clf):
        self.val = val
        clf = LogisticClassifier(clf)
        self.kernel = KDEGaussian(val, clf)

    def loss(self, x, y=0):
        x = as_tensor(x)
        dist = self.kernel.get_dist(y)
        return dist(x).item()


loss = LikelihoodLoss(val=val, clf=clf)

fig = CFigure(width=2 * width, height=height)
fig.subplot(1, 2, 1)
fig.sp.plot_fun(
    pois_attack.objective_function,
    plot_levels=False,
    colorbar=False,
    multipoint=False,
    n_grid_points=50,
    grid_limits=[(lb, ub), (lb, ub)],
)
# fig.sp.plot_fgrads(pois_attack.objective_function_gradient,
#                   n_grid_points=20, grid_limits=[(lb, ub), (lb, ub)])
fig.sp.plot_decision_regions(
    clf, plot_background=False, n_grid_points=500, grid_limits=[(lb, ub), (lb, ub)]
)
fig.sp.grid(grid_on=True, linestyle=":", linewidth=0.5)
fig.sp.plot_ds(tr, markers=".")
# fig.sp.plot_path(pois_attack.x_seq)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect("equal")
fig.sp.title("Bi-level Objective")

fig.subplot(1, 2, 2)
fig.sp.plot_fun(
    loss.loss,
    plot_levels=False,
    colorbar=False,
    multipoint=False,
    n_grid_points=50,
    grid_limits=[(lb, ub), (lb, ub)],
)
# fig.sp.plot_fgrads(pois_attack._objective_function_gradient,
#                   n_grid_points=20, grid_limits=[(-20, 20), (-20, 20)])
fig.sp.plot_decision_regions(
    clf, plot_background=False, n_grid_points=500, grid_limits=[(lb, ub), (lb, ub)]
)
fig.sp.grid(grid_on=True, linestyle=":", linewidth=0.5)
fig.sp.plot_ds(tr, markers=".")
# fig.sp.plot_path(pois_attack.x_seq)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect("equal")
fig.sp.title("Our Objective")

fig.savefig("poison_objectives.png")
fig.close()
