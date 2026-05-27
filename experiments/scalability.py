import argparse
import time
import numpy as np

import insight_common as ic

SIZES = [2500, 5000, 10000, 20000]      # sizes where PAM is still feasible


def subsample(X, n, seed):
    rng = np.random.RandomState(seed)
    return X[rng.choice(len(X), min(n, len(X)), replace=False)]


def time_fit(algorithm, X, k, seed):
    ic.configure_mixed_distance([2, 3, 4], 2.0)        # BankSim categoricals
    model = ic.make_clusterer(algorithm, k, "mixed", seed)
    t0 = time.time()
    model.fit(X)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    ap.add_argument("--k", type=int, default=17)
    ap.add_argument("--pam-cap", type=int, default=20000,
                    help="largest N at which PAM/CLARANS are attempted")
    args = ap.parse_args()

    print("=" * 70)
    print("SCALABILITY  (manuscript Table tab:thm6-validation, BankSim)")
    print("=" * 70)
    data = ic.load_dataset2(args.data_dir, args.seed)
    Xfull = data["X_train"]
    results = {a: [] for a in ("pam", "clara", "clarans")}

    for n in SIZES:
        X = subsample(Xfull, n, args.seed)
        print(f"\nN = {len(X)}")
        for algo in ("pam", "clara", "clarans"):
            if algo in ("pam", "clarans") and len(X) > args.pam_cap:
                results[algo].append(None)
                print(f"  {algo:9s}: skipped (N>{args.pam_cap}, O(N^2))")
                continue
            try:
                t = time_fit(algo, X, args.k, args.seed)
                results[algo].append(t)
                print(f"  {algo:9s}: {t:8.2f} s")
            except Exception as exc:                   # noqa: BLE001
                results[algo].append(None)
                print(f"  {algo:9s}: failed ({exc})")

    # CLARA additionally on the full training set
    print(f"\nN = {len(Xfull)} (full BankSim training set)")
    try:
        t_full = time_fit("clara", Xfull, args.k, args.seed)
        print(f"  clara    : {t_full:8.2f} s")
    except Exception as exc:                           # noqa: BLE001
        t_full = None
        print(f"  clara    : failed ({exc})")

    print("\n" + "=" * 70)
    print("LaTeX rows for tab:thm6-validation:")
    print("=" * 70)
    names = {"pam": "INSIGHT-PAM", "clara": "INSIGHT-CLARA",
             "clarans": "INSIGHT-CLARANS"}
    header = " & ".join(str(s) for s in SIZES)
    print(f"Variant & {header} \\\\")
    for algo in ("pam", "clara", "clarans"):
        cells = [names[algo]] + [ic.fmt(t, 2) if t is not None else "-"
                                 for t in results[algo]]
        print(ic.latex_row(cells))
    print(f"\nCLARA on full BankSim ({len(Xfull)} pts): "
          f"{ic.fmt(t_full, 2)} s")


if __name__ == "__main__":
    main()
