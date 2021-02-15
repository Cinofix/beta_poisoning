from secml.array import CArray
from src.classifier.secml_autograd import as_tensor
import torch
import torch.nn as nn


class GKDE(nn.Module):
    def __init__(self, data, bw):
        super().__init__()
        self.data = data
        self.bw = bw

    def forward(self, x):
        n = self.data.shape[0]
        num = -(self.data - x).norm(2, dim=1) / self.bw
        p = torch.exp(num).sum()
        p *= 1 / n
        return p


class GaussianOnFeatureSpace:
    def __init__(self, ds, clf, store_ds=False):
        self.data = ds.X.tondarray()
        self.labels = ds.Y.unique()
        self.clf = clf
        self.stat_register = {}
        self.ds_register = {}

        ylst = ds.Y.unique()
        for y in ylst:
            # embed data point with phi function
            ds_y = self.clf.transform_all(ds[ds.Y == y, :].X)
            if isinstance(ds_y, CArray):
                ds_y = as_tensor(ds_y).to(clf.device)
            # get stats on phi space
            mu = self._mu(ds_y)
            sigma = self._sigma(ds_y)
            self.stat_register[str(y)] = {"mu": mu, "sigma": sigma}
            if store_ds:
                h = torch.cdist(ds_y, ds_y).mean()
                self.ds_register[str(y)] = GKDE(ds_y, bw=h)

    def _mu(self, data):
        return data.mean(axis=0)

    def _sigma(self, data):
        return data.var(axis=0)


class KDEGaussian(GaussianOnFeatureSpace):
    def __init__(self, ds, clf):
        super().__init__(ds, clf, store_ds=True)

    def pr(self, x, y=None):
        kde = self.get_dist(y)
        phi_x = self.clf.transform(x)
        p = kde(phi_x)
        return p

    def get_dist(self, y):
        key = str(y)
        return self.ds_register[key]
