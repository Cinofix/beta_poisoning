import sys

sys.path.extend(["./"])

import csv, ray, time
from src.experiments.utilities import set_seed
from src.classifier.clf_utilities import test_poison
from src.optimizer.beta_optimizer import to_scaled_img

ray.init()


@ray.remote(num_cpus=0.75)
def f_pois(p_generator, clf, ds, box, n_poison, k):
    tr, val, ts = ds

    print("Ready to run poisoning attack")
    start = time.time()
    x_poison, y_poison = p_generator(
        tr=tr,
        val=val,
        clf=clf,
        box=box,
        n_poisoning_points=n_poison,
        k=k,
        transform=to_scaled_img,
    )
    time_required = time.time() - start
    clf_p, test_acc, val_acc = test_poison(clf, tr, val, ts, x_poison, y_poison)

    return (
        val_acc,
        test_acc,
        time_required,
    )


def run_ablation(p_generator, clf, c, ds, name, box, proto_lst=None, n_poison=100):
    set_seed(444)

    with open(name + ".csv", "w") as f:
        writer = csv.writer(
            f, escapechar=" ", quoting=csv.QUOTE_NONE, quotechar="", delimiter=","
        )
        writer.writerow(["clf,c,n_proto"] + ["val_acc,test_acc"] + ["time"])
        for n_proto in proto_lst:
            ret_id = []
            # put input data in shared memory
            ds_id = ray.put(ds)

            for i in range(5):
                id_remote = f_pois.remote(
                    p_generator, clf, ds_id, box, n_poison, n_proto
                )
                ret_id.append(id_remote)

            results = ray.get(ret_id)
            for i, ret in enumerate(results):
                val_acc, test_acc, time_req = ret
                writer.writerow(
                    [clf.to_string(), c, n_proto, val_acc, test_acc, time_req]
                )

                print(
                    "n_proto = %d c= %d, test_acc = %.3f time = %s"
                    % (n_proto, c, test_acc, time_req)
                )
            f.flush()
        f.close()
