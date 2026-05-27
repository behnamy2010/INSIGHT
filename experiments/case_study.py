import argparse
import numpy as np

import insight_common as ic


def profile_clusters(data, k_range, seed):
    from sklearn.metrics import silhouette_samples

    k, _, _, model = ic.run_insight(
        data, algorithm="clara", metric="mixed", oversampling="cluster",
        k_range=k_range, seed=seed)
    X, y = data["X_train"], data["y_train"].ravel()
    lab = model.predict(X)
    centroids = model.cluster_centers_
    ic.configure_mixed_distance(data["cat_cols"], 2.0)

    sample = X[:20000] if len(X) > 20000 else X
    sample_lab = lab[:20000] if len(X) > 20000 else lab
    try:
        sil = silhouette_samples(sample, sample_lab)
        sil_by_cluster = {c: sil[sample_lab == c].mean()
                          for c in range(k) if (sample_lab == c).any()}
    except Exception:                                  # noqa: BLE001
        sil_by_cluster = {c: float("nan") for c in range(k)}

    rows = []
    for c in range(k):
        mask = lab == c
        if mask.sum() == 0:
            continue
        d2med = ic.mixed_distance(X[mask], centroids[c][None, :])[:, 0].mean()
        rows.append(dict(cluster=c, size=int(mask.sum()),
                         fraud=float(y[mask].mean()),
                         silhouette=sil_by_cluster.get(c, float("nan")),
                         dist_medoid=float(d2med),
                         mean_features=X[mask].mean(axis=0)))
    return rows, k


def describe(data, rows):
    feat = data["feature_names"]
    rows_sorted = sorted(rows, key=lambda r: r["fraud"], reverse=True)
    top = rows_sorted[0]
    print(f"\n  Highest-fraud cluster: id={top['cluster']}  "
          f"size={top['size']}  fraud={100*top['fraud']:.1f}%  "
          f"silhouette={top['silhouette']:.3f}  "
          f"mean-dist-medoid={top['dist_medoid']:.2f}")
    print("  Mean feature values (this cluster):")
    for name, val in zip(feat, top["mean_features"]):
        print(f"    {name:14s} {val:12.4f}")
    # contrast with a clean, legitimate-dominant, cohesive cluster
    clean = sorted([r for r in rows if r["fraud"] < 0.01],
                   key=lambda r: r["silhouette"], reverse=True)
    if clean:
        cl = clean[0]
        print(f"  Contrast cluster (legitimate-dominant): id={cl['cluster']} "
              f"size={cl['size']}  fraud={100*cl['fraud']:.2f}%  "
              f"silhouette={cl['silhouette']:.3f}")
    return top


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 40)

    print("=" * 70)
    print("INTERPRETABILITY CASE STUDIES  (manuscript Section 5)")
    print("=" * 70)

    print("\n[Case study 1: European Credit Card -- Cluster 12 analogue]")
    d1 = ic.load_dataset1(args.data_dir, args.seed)
    rows1, _ = profile_clusters(d1, k_range, args.seed)
    describe(d1, rows1)

    print("\n[Case study 2: BankSim -- high-fraud merchant cluster]")
    d2 = ic.load_dataset2(args.data_dir, args.seed)
    rows2, _ = profile_clusters(d2, k_range, args.seed)
    top = describe(d2, rows2)
    print("\n  -> Use the id, size, fraud rate, and the category_ID / amount / "
          "age / gender means above to fill the \\runres{...} entries in "
          "case study 2 of Section 5. (category_ID and customer/merchant IDs "
          "map back to names via the groupby tables in notebook-dataset2.)")


if __name__ == "__main__":
    main()
