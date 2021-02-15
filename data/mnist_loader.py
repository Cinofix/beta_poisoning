from secml.data.splitter import CTrainTestSplit
from secml.data.loader import CDataLoaderMNIST


def get_mnist_loader(num_samples, mode="training", digits=tuple(range(10))):
    loader = CDataLoaderMNIST()
    tr_data = loader.load(mode, digits=digits, num_samples=num_samples)
    return tr_data


def get_mnist(digits=tuple(range(10)), n_tr=None, n_ts=None):
    # dataset creation
    train = get_mnist_loader(mode="training", digits=digits, num_samples=n_tr)
    test = get_mnist_loader(mode="testing", digits=digits, num_samples=n_ts)
    # normalize data
    train.X /= 255.0
    test.X /= 255.0
    return train, test


def load_mnist(digits=tuple(range(10)), n_tr=100, n_val=100, n_ts=200, random_state=12):
    train, ts = get_mnist(digits=digits, n_tr=n_tr + n_val, n_ts=n_ts)
    # Split in training, validation and experiments
    if n_val > 0:
        splitter = CTrainTestSplit(
            train_size=n_tr, test_size=n_val, random_state=random_state
        )
        tr, val = splitter.split(train)
        return tr, val, ts
    return train, ts
