import torch
import numpy as np


def rand_flipper(y_base, y):
    return np.random.choice(a=y_base[y_base != y], size=1)[0]


def max_loss_flipper(loss_map, y):
    return loss_map[y].item()


def pick_poison_label(val, y_from, y_poison=None):
    if y_poison is None:
        # if target labels are not given then all the labels are candidates
        y_poison = val.Y.unique().tondarray()
    y_poison_chosen = [rand_flipper(y_poison, y) for y in y_from]
    return np.array(y_poison_chosen)


def clamp_to_box(x, box):
    return torch.clamp(x, box[0], box[1])


def clamp_to_convex(betas):
    betas = betas.clamp(-1, 1)
    return betas / (betas.sum() + 1e-32)


def at_least_dimension(x, k, pos=-1):
    for _ in range(k):
        x = x.unsqueeze(pos)
    return x


def pick_random_protos(ds, y, k=10):
    samples_in_y = ds.X[ds.Y == y, :]
    samples_in_y.shuffle()
    return samples_in_y[:k, :].tondarray()


def to_scaled_img(x):
    x = x.cpu().numpy().flatten()
    return x.clip(0, 1)


def get_poison_target_labels(val, n_poison, y_target, y_poison):
    if y_target is None:
        y_target = val.Y.unique().tondarray()

    y_target_chosen = np.random.choice(a=y_target, size=n_poison, replace=True)

    y_poison_chosen = pick_poison_label(val, y_poison=y_poison, y_from=y_target_chosen)
    return y_target_chosen, y_poison_chosen
