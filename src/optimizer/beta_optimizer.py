from src.optimizer.attack_utilities import *
from src.optimizer.kde import KDEGaussian


def optimizer(clf, x_proto, y_target, box, kernel=None, lb=1):
    x_proto_p = clf.preprocess(x_proto).to(clf.device).squeeze()
    n = x_proto_p.shape[0]

    betas = torch.rand(n).to(clf.device)
    betas = at_least_dimension(betas, len(x_proto_p.shape) - 1).requires_grad_()
    # optimizer for the adversarial sample
    adversarial_optimizer = torch.optim.SGD([betas], lr=1e-2)

    opt_loss = float("inf")
    opt_beta = betas.clone()
    dist = float("inf")
    pr_before = 0

    while dist > 1e-05:
        adversarial_optimizer.zero_grad()
        poison_sample = (betas * x_proto_p).sum(axis=0).unsqueeze(0)
        poison_sample = clamp_to_box(poison_sample, box)

        pr_loss = kernel.pr(poison_sample, y_target)
        loss = -lb * pr_loss

        dist = (pr_loss - pr_before).item()
        pr_before = pr_loss
        if loss < opt_loss:
            opt_beta = betas.data.detach().clone()
            opt_loss = loss

        loss.backward()
        adversarial_optimizer.step()

        # ensuring that the image is valid
        betas.data = clamp_to_convex(betas.data)
        poison_sample.data = clamp_to_box(poison_sample.data, box)
    x_poison = (opt_beta * x_proto_p).sum(axis=0)
    x_poison = clamp_to_box(x_poison, box).detach()
    return x_poison


def run_beta_attack(
    val,
    clf,
    y_target,
    y_poison,
    n_poison,
    box,
    lb=1,
    k=None,
    transform=to_scaled_img,
    verbose=False,
):
    n, m = val.X.shape
    kernel = KDEGaussian(val, clf)
    x_poison = np.zeros((n_poison, m), dtype=val.X.dtype)
    x_proto = np.zeros((n_poison, k, m), dtype=val.X.dtype)

    for i in range(n_poison):

        if verbose:
            print("Opt: ", i)

        x_proto[i] = pick_random_protos(val, y_target[i], k=k)
        x_p = optimizer(
            clf=clf,
            x_proto=x_proto[i],
            y_target=y_target[i],
            box=box,
            lb=lb,
            kernel=kernel,
        )
        x_poison[i] = transform(x_p)

    return x_poison, x_proto


def beta_poison(
    tr,
    val,
    clf,
    box,
    n_poisoning_points,
    lb=1,
    k=None,
    y_target=None,
    y_poison=None,
    verbose=False,
    transform=to_scaled_img,
):

    # the attacker specify the target classes
    y_target_chosen, y_poison_chosen = get_poison_target_labels(
        val, n_poisoning_points, y_target, y_poison
    )

    x_poison, x_proto = run_beta_attack(
        val=val,
        clf=clf,
        y_target=y_target_chosen,
        y_poison=y_poison_chosen,
        n_poison=n_poisoning_points,
        box=box,
        k=k,
        lb=lb,
        verbose=verbose,
        transform=transform,
    )
    return x_poison, y_poison_chosen
