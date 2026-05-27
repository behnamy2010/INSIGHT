import argparse
import numpy as np

import insight_common as ic


def build_insight_predictor(data, k_range, seed):
    """Train INSIGHT-CLARA and return its fraud-probability prediction function.

    Returns (predict_fraud, info) where predict_fraud maps an (n, d) array to an
    (n,) array of fraud probabilities, exactly as INSIGHT-CLARA scores them.
    """
    k, test, _, model = ic.run_insight(
        data, algorithm="clara", metric="mixed", oversampling="cluster",
        k_range=k_range, seed=seed)
    lab_tr = model.predict(data["X_train"])
    y_tr = data["y_train"].ravel()
    freq = np.array([(y_tr[lab_tr == c].mean() if (lab_tr == c).any() else 0.0)
                     for c in range(k)])

    def predict_fraud(X):
        return freq[model.predict(np.asarray(X, dtype=float))]

    return predict_fraud, dict(k=k, test=test, freq=freq)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    ap.add_argument("--dataset", choices=["1", "2"], default="1")
    ap.add_argument("--n-background", type=int, default=200)
    ap.add_argument("--n-explain", type=int, default=400)
    args = ap.parse_args()
    k_range = range(2, 8) if args.quick else range(2, 40)

    print("=" * 70)
    print("SHAP INTERPRETATION OF INSIGHT  (manuscript Section 5.6)")
    print("=" * 70)
    loader = ic.load_dataset1 if args.dataset == "1" else ic.load_dataset2
    data = loader(args.data_dir, args.seed)

    print(f"[{data['name']}] training INSIGHT-CLARA ...")
    predict_fraud, info = build_insight_predictor(data, k_range, args.seed)
    print(f"  INSIGHT-CLARA: k={info['k']}, test F1={info['test']['f1']:.4f}")

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"shap/matplotlib not available: {exc}")
        return

    rng = np.random.RandomState(args.seed)
    bg = data["X_train"][rng.choice(len(data["X_train"]),
                                    min(args.n_background, len(data["X_train"])),
                                    replace=False)]
    ex = data["X_test"][rng.choice(len(data["X_test"]),
                                   min(args.n_explain, len(data["X_test"])),
                                   replace=False)]

    print(f"  running KernelExplainer (background={len(bg)}, explain={len(ex)}) ...")
    explainer = shap.KernelExplainer(predict_fraud, bg)
    shap_values = explainer.shap_values(ex, nsamples=400)

    shap.summary_plot(shap_values, ex, feature_names=data["feature_names"],
                      show=False)
    plt.tight_layout()
    out = f"shap_insight_dataset{args.dataset}.png"
    plt.savefig(out, dpi=200)
    print(f"\nSHAP summary plot saved to {out}")
    print("This is the plot for manuscript Figure fig:shap_plot ")

    # mean |SHAP| ranking, for the text of Section 5.6
    order = np.argsort(-np.abs(shap_values).mean(axis=0))
    print("\nFeature importance ranking (mean |SHAP|):")
    for r in order[:10]:
        print(f"  {data['feature_names'][r]:14s} "
              f"{np.abs(shap_values).mean(axis=0)[r]:.5f}")


if __name__ == "__main__":
    main()
