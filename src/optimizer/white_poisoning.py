from secml.adv.attacks import CAttackPoisoningSVM, CAttackPoisoningLogisticRegression

solver_params = {
    "eta": 0.5,
    "eta_min": 2.0,
    "eta_max": None,
    "max_iter": 150,
    "eps": 1e-6,
}


def white_poison(tr, val, clf, box, n_poisoning_points, **options):

    if "svm" in clf.to_string():
        pois_attack = CAttackPoisoningSVM(
            classifier=clf.clf,
            training_data=tr,
            val=val,
            lb=box[0],
            ub=box[1],
            solver_params=solver_params,
        )
    else:  # white-box available only for svm and logistic
        pois_attack = CAttackPoisoningLogisticRegression(
            classifier=clf.clf,
            training_data=tr,
            val=val,
            lb=box[0],
            ub=box[1],
            solver_params=solver_params,
        )
    pois_attack.n_points = n_poisoning_points
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(val.X, val.Y)
    return pois_ds.X, pois_ds.Y
