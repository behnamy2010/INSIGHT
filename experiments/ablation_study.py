import argparse
import numpy as np

import insight_common as ic
from alpha_ablation import estimate_data_driven_alpha

# (label, distance, alpha-mode, oversampling, algorithm)
CONFIGS = [
    ("Baseline-1", "onehot", 2.0,  "none",    "clara"),
    ("Baseline-2", "onehot", 2.0,  "global",  "clara"),
    ("Baseline-3", "gower",  2.0,  "none",    "clara"),
    ("Baseline-4", "gower",  2.0,  "global",  "clara"),
    ("Ablation A", "mixed",  "dd", "none",    "clara"),
    ("Ablation B", "mixed",  "dd", "global",  "clara"),
    ("Ablation C", "mixed",  "dd", "cluster", "pam"),
    ("Ablation E", "mixed",  "dd", "cluster", "clarans"),
    ("INSIGHT (full)", "mixed", "dd", "cluster", "clara"),
]


def algo_subsample(data, cap_minority=1000, total=3000, seed=42):
    """Stratified sub-sample for the PAM/CLARANS rows (PAM is O(N^2))."""
    rng = np.random.RandomState(seed)
    y = data["y_train"].ravel()
    mino = np.where(y == 1)[0]
    majo = np.where(y == 0)[0]
    take_min = min(len(mino), cap_minority)
    take_maj = min(len(majo), total - take_min)
    idx = np.concatenate([rng.choice(mino, take_min, replace=False),
                          rng.choice(majo, take_maj, replace=False)])
    rng.shuffle(idx)
    out = dict(data)
    out["X_train"] = data["X_train"][idx]
    out["y_train"] = data["y_train"][idx]
    return out


def run_grid(data, k_range, seed):
    alpha_dd = (estimate_data_driven_alpha(data, seed=seed)[0]
                if data["cat_cols"] else 2.0)
    # Beta defaults to 1.0; INSIGHT-full on BankSim uses beta=2.0, the value
    # selected by the joint validation sweep (see joint_alpha_beta_sweep.py).
    # PAM and CLARANS rows are left at beta=1.0 (they are run on a sub-sample
    # and re-running them at the larger beta was deemed too costly).
    rows = {}
    for label, dist, amode, oversampling, algo in CONFIGS:
        metric = "gower" if dist == "gower" else "mixed"
        alpha = alpha_dd if amode == "dd" else 2.0
        d = algo_subsample(data, seed=seed) if algo in ("pam", "clarans") else data
        if (data["cat_cols"] and algo == "clara"
                and oversampling == "cluster" and label.startswith("INSIGHT")):
            beta = 2.0
        else:
            beta = 1.0
        os_rate = beta * 0.02
        try:
            k, test, secs, _ = ic.run_insight(
                d, algorithm=algo, metric=metric, oversampling=oversampling,
                k_range=k_range, alpha=alpha, os_rate=os_rate, seed=seed)
            rows[label] = (test["f1"], test["mcc"])
            print(f"  {label:16s} dist={dist:7s} a={alpha:6.3f} os={oversampling:7s} "
                  f"algo={algo:8s} beta={beta:.1f} -> F1={test['f1']:.5f} "
                  f"MCC={test['mcc']:.5f} (k={k}, {secs:.1f}s)", flush=True)
        except Exception as exc:                       # noqa: BLE001
            rows[label] = None
            print(f"  {label:16s} FAILED: {exc}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 22)

    print("=" * 70)
    print("ABLATION STUDY  (manuscript Table tab:ablation; validation k-sweep)")
    print("=" * 70)
    print("\n[Dataset 1: European Credit Card]", flush=True)
    d1 = ic.load_dataset1(args.data_dir, args.seed)
    r1 = run_grid(d1, k_range, args.seed)
    print("\n[Dataset 2: BankSim]", flush=True)
    d2 = ic.load_dataset2(args.data_dir, args.seed)
    r2 = run_grid(d2, k_range, args.seed)

    print("\n" + "=" * 70)
    print("LaTeX rows for tab:ablation (F1/MCC D1, F1/MCC D2):")
    print("=" * 70)
    for label, *_ in CONFIGS:
        a, b = r1.get(label), r2.get(label)
        cells = [label]
        for res in (a, b):
            cells += (["-", "-"] if res is None
                      else [ic.fmt(res[0]), ic.fmt(res[1])])
        print(ic.latex_row(cells))


if __name__ == "__main__":
    main()
