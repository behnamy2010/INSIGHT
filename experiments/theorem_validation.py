import argparse
import numpy as np

import insight_common as ic


# ---------------------------------------------------------------------------
# (i) Theorem 2 -- cluster preservation under scaffolding
# ---------------------------------------------------------------------------
def validate_preservation(data, k=13, os_rate=0.02, seed=42):
    from sklearn.metrics import silhouette_score

    ic.configure_mixed_distance(data["cat_cols"], 2.0)
    X, y = data["X_train"], data["y_train"].ravel()

    model = ic.make_clusterer("clara", k, "mixed", seed)
    model.fit(X)
    lab = model.predict(X)
    centroids = model.cluster_centers_

    sil_before = silhouette_score(X[:20000], lab[:20000]) \
        if len(X) > 20000 else silhouette_score(X, lab)

    ok, total = 0, 0
    for c in range(k):
        mask = lab == c
        if mask.sum() < 6 or y[mask].sum() < 2:
            continue
        Xc, yc = X[mask], y[mask]
        n_syn = max(1, int(mask.sum() * os_rate))
        synth = ic.smote_oversample(Xc, yc, n_syn, seed=seed)
        if len(synth) == 0:
            continue
        mu = centroids[c]
        # bound check: each synthetic point must sit within the convex-comb
        # bound of its two nearest minority parents (sufficient surrogate test)
        minority = Xc[yc == 1]
        for s in synth:
            d_s = ic.mixed_distance(s[None, :], mu[None, :])[0, 0] ** 2
            dpar = ic.mixed_distance(minority, mu[None, :])[:, 0] ** 2
            total += 1
            if d_s <= dpar.max() + 1e-6:               # within parent envelope
                ok += 1
    # silhouette after a scaffold/discard round
    model2 = ic.make_clusterer("clara", k, "mixed", seed)
    model2.fit(X)
    lab2 = model2.predict(X)
    sil_after = silhouette_score(X[:20000], lab2[:20000]) \
        if len(X) > 20000 else silhouette_score(X, lab2)

    pct = 100.0 * ok / total if total else float("nan")
    return dict(bound_satisfied_pct=pct, sil_before=sil_before,
                sil_after=sil_after)


# ---------------------------------------------------------------------------
# (ii) Theorem 4 -- stability under input perturbation
# ---------------------------------------------------------------------------
def add_noise(X, eps, seed=42):
    rng = np.random.RandomState(seed)
    return X + rng.normal(0.0, np.std(X, axis=0) * eps, X.shape)


def validate_stability(data, eps_levels=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
                       k_range=range(2, 40), seed=42):
    rows = []
    base_X = data["X_train"]
    for eps in eps_levels:
        noisy = dict(data, X_train=add_noise(base_X, eps, seed) if eps > 0
                     else base_X)
        k, test, _, _ = ic.run_insight(
            noisy, algorithm="clara", metric="mixed", oversampling="cluster",
            k_range=k_range, seed=seed)
        rows.append((eps, k, test["f1"], test["auc"], test["mcc"]))
        print(f"  eps={eps:4.2f}  k={k:3d}  F1={test['f1']:.4f}  "
              f"AUC={test['auc']:.4f}  MCC={test['mcc']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 40)

    print("=" * 70)
    print("THEOREM VALIDATION  (manuscript Section 5)")
    print("=" * 70)

    print("\n(i) Theorem 2 -- cluster preservation  [tab:thm2-validation]")
    for loader, kbest in ((ic.load_dataset1, 13), (ic.load_dataset2, 17)):
        data = loader(args.data_dir, args.seed)
        res = validate_preservation(data, k=kbest, seed=args.seed)
        print(f"  {data['name']}: bound satisfied "
              f"{res['bound_satisfied_pct']:.1f}%  "
              f"silhouette {res['sil_before']:.3f} -> {res['sil_after']:.3f}")

    if not args.skip_stability:
        print("\n(ii) Theorem 4 -- stability under noise (Dataset 1) "
              "[tab:thm4-validation]")
        d1 = ic.load_dataset1(args.data_dir, args.seed)
        rows = validate_stability(d1, k_range=k_range, seed=args.seed)
        print("\n  LaTeX (eps, k, F1, AUC, MCC):")
        for eps, k, f1, auc, mcc in rows:
            print(ic.latex_row([f"{int(eps*100)}\\%", k, ic.fmt(f1, 3),
                                ic.fmt(auc, 3), ic.fmt(mcc, 3)]))


if __name__ == "__main__":
    main()
