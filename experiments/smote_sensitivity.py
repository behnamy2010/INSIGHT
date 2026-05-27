import argparse
import time

import insight_common as ic


def beta_sweep(data, betas, k_range, seed):
    out = []
    for beta in betas:
        # os_rate in run_insight plays the role of the oversampling ratio beta
        k, test, _, _ = ic.run_insight(
            data, algorithm="clara", metric="mixed", oversampling="cluster",
            k_range=k_range, os_rate=beta * 0.02, seed=seed)
        out.append((beta, k, test["f1"]))
        print(f"  beta={beta:4.2f} -> val-selected k={k}, test F1={test['f1']:.5f}")
    return out


def overhead(data, k_range, seed):
    t0 = time.time()
    ic.run_insight(data, algorithm="clara", metric="mixed",
                   oversampling="cluster", k_range=k_range, seed=seed)
    t_cluster = time.time() - t0
    t0 = time.time()
    ic.run_insight(data, algorithm="clara", metric="mixed",
                   oversampling="global", k_range=k_range, seed=seed)
    t_global = time.time() - t0
    return t_cluster, t_global


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 40)
    betas = [0.5, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("CLUSTER-AWARE SMOTE SENSITIVITY  (Table tab:smote-sensitivity)")
    print("=" * 70)

    results = {}
    for loader in (ic.load_dataset1, ic.load_dataset2):
        data = loader(args.data_dir, args.seed)
        print(f"\n[{data['name']}] beta sweep")
        sweep = beta_sweep(data, betas, k_range, args.seed)
        print(f"[{data['name']}] overhead (cluster-aware vs. global SMOTE)")
        tc, tg = overhead(data, k_range, args.seed)
        print(f"  cluster-aware: {tc:.2f} s   global: {tg:.2f} s")
        results[data["name"]] = (sweep, tc, tg)

    print("\n" + "=" * 70)
    print("LaTeX for tab:smote-sensitivity:")
    print("=" * 70)
    for name, (sweep, tc, tg) in results.items():
        f1s = " & ".join(ic.fmt(f1) for _, _, f1 in sweep)
        print(f"Validation F1 ({name}) & {f1s} \\\\")
    for name, (sweep, tc, tg) in results.items():
        print(f"Overhead ({name}): {tc:.2f} s cluster-aware vs. {tg:.2f} s global")


if __name__ == "__main__":
    main()
