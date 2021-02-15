from src.optimizer.attack_utilities import *


def pick_proto_(val, n_proto, y_target):
    x_target = val.X[val.Y == y_target, :]
    nt = x_target.shape[0]
    proto_index = list(np.random.choice(nt, n_proto, replace=False))
    return x_target[proto_index, :].tondarray(), proto_index


def pick_proto_from_target(val, y_target, n_poisoning_points, n_proto):
    x_proto_i = []
    for i in range(n_poisoning_points):
        prototypes, _ = pick_proto_(val, n_proto, y_target[i])
        x_proto_i += [prototypes]
    x_prototypes = np.stack(x_proto_i, axis=0)
    return x_prototypes


def flip_batch_poison(tr, val, clf, box, n_poison, **params):
    n, m = val.X.shape
    y_target = params["y_target"]
    y_poison = params["y_poison"]
    y_target_chosen = np.random.choice(a=y_target, size=n_poison, replace=True)
    y_poison_chosen = pick_poison_label(val, y_poison=y_poison, y_from=y_target_chosen)
    x_proto = pick_proto_from_target(val, y_target_chosen, n_poison, 1)
    x_poison = x_proto.reshape(-1, m)
    return x_poison, y_poison_chosen
