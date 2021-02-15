from secml.array import CArray
from src.classifier.clf_utilities import test_clf, test_poison
from src.experiments.utilities import *
import csv


def write(
    tr,
    algorithm,
    lb,
    n_poison,
    c,
    val_acc,
    test_acc,
    time,
    writer,
):
    tr_size = length(tr)

    writer.writerow(
        [
            algorithm,
            lb,
            tr_size,
            n_poison,
            n_poison / (tr_size + n_poison),
            c,
            val_acc,
            test_acc,
            time,
        ]
    )

    print(
        "C = %s, fr_pois %f, acc= %.3f, time=%f"
        % (
            c,
            n_poison / (tr_size + n_poison),
            test_acc,
            time,
        )
    )


def write_bin(rs, tr, val, ts, clf, c, algorithm, time, params, writer):
    x_poison, y_poison = rs
    if isinstance(x_poison, CArray):
        n_poison = x_poison.shape[0]
    else:
        n_poison = len(x_poison)

    lb = params["lb"]

    if n_poison == 0:
        clf_p, test_acc, val_acc = clf, test_clf(clf, ts), test_clf(clf, val)
    else:
        clf_p, test_acc, val_acc = test_poison(clf, tr, val, ts, x_poison, y_poison)

    write(
        tr, algorithm, lb, n_poison, c, val_acc, test_acc, time, writer,
    )


def run_attack(generator, path, clf, tr, val, ts, params):
    set_seed(444)
    box = (val.X.min(), val.X.max())
    n_poisoning_points = np.linspace(start=0.05, stop=0.25, num=10) * tr.Y.size
    g_name = path.split("/")[-1]

    with open(path + ".csv", "w") as file:
        writer = csv.writer(
            file, escapechar=" ", quoting=csv.QUOTE_NONE, quotechar="", delimiter=","
        )
        writer.writerow(
            [
                "algorithm,lb,tr_size,n_poison,poison_fraction,"
                "c,val_acc,test_acc,"
                "time"
            ]
        )

        for _ in range(5):
            for c in [1, 100]:
                start = time.time()
                clf.init_fit(tr, parameters={"C": c})
                training_time = time.time() - start

                if params["y_target"] is None:
                    params["y_target"] = val.Y.unique().tondarray()
                empty_attack = ([], [])

                write_bin(
                    empty_attack,
                    tr,
                    val,
                    ts,
                    clf,
                    c,
                    g_name,
                    training_time,
                    params,
                    writer,
                )

                for n_poison in n_poisoning_points.astype(int):
                    start = time.time()
                    x_poison, y_poison = generator(
                        tr,
                        val,
                        clf,
                        box,
                        n_poison,
                        lb=params["lb"],
                        k=params["n_proto"],
                        y_target=params["y_target"],
                        y_poison=params["y_poison"],
                        verbose=False,
                        transform=params["transform"],
                    )

                    exec_time = time.time() - start
                    res = [x_poison, y_poison]
                    write_bin(
                        res, tr, val, ts, clf, c, g_name, exec_time, params, writer,
                    )
                    file.flush()
        file.close()
