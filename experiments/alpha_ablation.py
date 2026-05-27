import argparse
import numpy as np

import insight_common as ic


def estimate_data_driven_alpha(data, n_pairs=20000, seed=42):
    """alpha = E[d_num^2] / E[delta_cat] over random training-point pairs."""
    rng = np.random.RandomState(seed)
    X = data["X_train"]
    cat = data["cat_cols"]
    num = [j for j in range(X.shape[1]) if j not in cat]
    i = rng.randint(0, len(X), n_pairs)
    j = rng.randint(0, len(X), n_pairs)
    d_num_sq = np.sum((X[i][:, num] - X[j][:, num]) ** 2, axis=1)
    delta_cat = np.sum(X[i][:, cat] != X[j][:, cat], axis=1)
    e_num = float(np.mean(d_num_sq))
    e_cat = float(np.mean(delta_cat))
    return e_num / e_cat if e_cat > 0 else 1.0, e_num, e_cat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 40)

    print("=" * 70)
    print("ALPHA ABLATION  (manuscript Table tab:alpha-ablation, BankSim)")
    print("=" * 70)
    data = ic.load_dataset2(args.data_dir, args.seed)

    alpha_dd, e_num, e_cat = estimate_data_driven_alpha(data, seed=args.seed)
    print(f"Data-driven alpha = E[d_num^2]/E[delta_cat] = {e_num:.4f}/{e_cat:.4f}"
          f" = {alpha_dd:.4f}\n")

    sweep = [("0.5", 0.5), ("1", 1.0), ("2 (one-hot equiv.)", 2.0),
             ("data-driven", alpha_dd)]
    rows = []
    for label, alpha in sweep:
        k, test, secs, _ = ic.run_insight(
            data, algorithm="clara", metric="mixed", oversampling="cluster",
            k_range=k_range, alpha=alpha, seed=args.seed)
        # validation F1 at the selected k for reporting
        rows.append((label, alpha, k, test["f1"], test["mcc"]))
        print(f"  alpha={label:20s} ({alpha:7.4f}) -> "
              f"test F1={test['f1']:.5f}  MCC={test['mcc']:.5f}  (k={k})")

    print("\n" + "=" * 70)
    print("LaTeX rows for tab:alpha-ablation:")
    print("=" * 70)
    for label, alpha, k, f1, mcc in rows:
        print(ic.latex_row([label, ic.fmt(alpha, 4),
                            "(val)", ic.fmt(f1), ic.fmt(mcc)]))


if __name__ == "__main__":
    main()
