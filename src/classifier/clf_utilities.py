from secml.ml.peval.metrics import CMetricAccuracy
from secml.data import CDataset


def test_clf(clf, ts):
    # Compute predictions on a experiments set
    y_pred = clf.predict(ts.X)
    # Metric to use for performance evaluation
    metric = CMetricAccuracy()
    # Evaluate the accuracy of the classifier
    acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
    return acc


def test_poison(clf, tr, val, ts, x_poison, y_poison):
    poison = CDataset(x_poison, y_poison)

    clf_p = clf.deepcopy()
    clf_p.init()
    tr_p = tr.append(poison)
    clf_p.fit(tr_p)

    test_acc = test_clf(clf_p, ts)
    val_acc = test_clf(clf_p, val)
    return clf_p, test_acc, val_acc
