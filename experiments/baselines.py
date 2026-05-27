
import argparse
import numpy as np

import insight_common as ic


def _threshold_by_f1(y_val, s_val, y_test, s_test):
    """Pick the decision threshold maximising validation F1, then score test."""
    from sklearn.metrics import f1_score
    grid = np.unique(np.quantile(s_val, np.linspace(0.01, 0.999, 200)))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (s_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return ic.evaluate(y_test, (s_test >= best_t).astype(float))


# --------------------------------------------------------------------------
# Supervised baselines
# --------------------------------------------------------------------------
def run_catboost(d):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import ParameterGrid
    spw = (d["y_train"] == 0).sum() / max(1, (d["y_train"] == 1).sum())
    grid = ParameterGrid({"depth": [4, 6, 8],
                          "iterations": [300, 600],
                          "learning_rate": [0.05, 0.1]})
    best, best_f1 = None, -1.0
    for p in grid:
        clf = CatBoostClassifier(**p, scale_pos_weight=spw,
                                 loss_function="Logloss", verbose=False,
                                 random_seed=42, thread_count=-1)
        clf.fit(d["X_train"], d["y_train"].ravel())
        f1 = ic.evaluate(d["y_val"], clf.predict_proba(d["X_val"])[:, 1])["f1"]
        if f1 > best_f1:
            best_f1, best = f1, clf
    return _threshold_by_f1(d["y_val"].ravel(),
                            best.predict_proba(d["X_val"])[:, 1],
                            d["y_test"].ravel(),
                            best.predict_proba(d["X_test"])[:, 1])


def _supervised_score(clf, d):
    """Fit and score with predict_proba if available, else decision_function."""
    clf.fit(d["X_train"], d["y_train"].ravel())

    def _s(X):
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[:, 1]
        if hasattr(clf, "decision_function"):
            return clf.decision_function(X)
        return clf.predict(X).astype(float)

    return _threshold_by_f1(d["y_val"].ravel(), _s(d["X_val"]),
                            d["y_test"].ravel(), _s(d["X_test"]))


# alias for backwards compatibility within this file
_supervised_proba = _supervised_score


def _scaled(d):
    """Return a copy of `d` with X_* standardised (for linear/MLP models)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(d["X_train"])
    return dict(d, X_train=sc.transform(d["X_train"]),
                X_val=sc.transform(d["X_val"]),
                X_test=sc.transform(d["X_test"]))


def run_random_forest(d):
    from sklearn.ensemble import RandomForestClassifier
    return _supervised_proba(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=42), d)


def run_logistic_regression(d):
    from sklearn.linear_model import LogisticRegression
    return _supervised_score(
        LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=42), _scaled(d))


def run_decision_tree(d):
    from sklearn.tree import DecisionTreeClassifier
    return _supervised_score(
        DecisionTreeClassifier(class_weight="balanced", random_state=42), d)


def run_knn(d):
    from sklearn.neighbors import KNeighborsClassifier
    return _supervised_score(KNeighborsClassifier(n_neighbors=5, n_jobs=-1), d)


def run_naive_bayes(d):
    from sklearn.naive_bayes import GaussianNB
    return _supervised_score(GaussianNB(), d)


def run_extra_trees(d):
    from sklearn.ensemble import ExtraTreesClassifier
    return _supervised_score(
        ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                             n_jobs=-1, random_state=42), d)


def run_adaboost(d):
    from sklearn.ensemble import AdaBoostClassifier
    return _supervised_score(
        AdaBoostClassifier(n_estimators=200, random_state=42), d)


def run_linear_svm(d):
    from sklearn.svm import LinearSVC
    return _supervised_score(
        LinearSVC(class_weight="balanced", max_iter=5000, dual="auto",
                  random_state=42), _scaled(d))


def run_lda(d):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return _supervised_score(LinearDiscriminantAnalysis(), _scaled(d))


def run_mlp(d):
    from sklearn.neural_network import MLPClassifier
    return _supervised_score(
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80,
                      early_stopping=True, random_state=42), _scaled(d))


# --------------------------------------------------------------------------
# Unsupervised baselines
# --------------------------------------------------------------------------
def run_isolation_forest(d):
    from sklearn.ensemble import IsolationForest
    contamination = float(np.clip(d["y_train"].mean(), 1e-4, 0.5))
    clf = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=42, n_jobs=-1)
    clf.fit(d["X_train"][d["y_train"].ravel() == 0])   # fit on legitimate only
    s_val = -clf.score_samples(d["X_val"])
    s_test = -clf.score_samples(d["X_test"])
    return _threshold_by_f1(d["y_val"].ravel(), s_val,
                            d["y_test"].ravel(), s_test)


def run_autoencoder(d):
    from tensorflow import keras
    from tensorflow.keras import layers
    X_norm = d["X_train"][d["y_train"].ravel() == 0]
    n_feat = X_norm.shape[1]
    enc_dim = max(2, n_feat // 2)
    model = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(enc_dim, activation="relu"),
        layers.Dense(max(2, enc_dim // 2), activation="relu"),
        layers.Dense(enc_dim, activation="relu"),
        layers.Dense(n_feat, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_norm, X_norm, epochs=20, batch_size=512, verbose=0,
              validation_split=0.1)

    def recon_err(A):
        return np.mean((A - model.predict(A, verbose=0)) ** 2, axis=1)

    return _threshold_by_f1(d["y_val"].ravel(), recon_err(d["X_val"]),
                            d["y_test"].ravel(), recon_err(d["X_test"]))


SUPERVISED = [("CatBoost", run_catboost),
              ("Random Forest", run_random_forest),
              ("Extra Trees", run_extra_trees),
              ("AdaBoost", run_adaboost),
              ("Logistic Regression", run_logistic_regression),
              ("Linear SVM", run_linear_svm),
              ("LDA", run_lda),
              ("MLP", run_mlp),
              ("Decision Tree", run_decision_tree),
              ("k-NN", run_knn),
              ("Naive Bayes", run_naive_bayes)]
UNSUPERVISED = [("Isolation Forest", run_isolation_forest),
                ("Deep Autoencoder", run_autoencoder)]


def run_for(data):
    rows = []
    for name, fn in SUPERVISED + UNSUPERVISED:
        try:
            m = fn(data)
            rows.append((name, m))
            print(f"  {name:20s} F1={m['f1']:.5f} MCC={m['mcc']:.5f} "
                  f"AUC={m['auc']:.5f}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            rows.append((name, None))
            print(f"  {name:20s} skipped: {exc}", flush=True)
    return rows


def emit(rows):
    for name, m in rows:
        if m is None:
            cells = ["This work", name] + ["-"] * 6
        else:
            cells = ["This work", name,
                     ic.fmt(m["accuracy"]), ic.fmt(m["precision"]),
                     ic.fmt(m["recall"]), ic.fmt(m["f1"]),
                     ic.fmt(m["auc"]), ic.fmt(m["mcc"])]
        print(ic.latex_row(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ic.add_arguments(ap)
    args = ap.parse_args()

    print("=" * 70)
    print("BASELINES  (manuscript Tables tab:comp-algs-others-1/2)")
    print("=" * 70)
    print("\n[Dataset 1: European Credit Card]", flush=True)
    d1 = ic.load_dataset1(args.data_dir, args.seed)
    r1 = run_for(d1)
    print("\n[Dataset 2: BankSim]", flush=True)
    d2 = ic.load_dataset2(args.data_dir, args.seed)
    r2 = run_for(d2)

    print("\n" + "=" * 70)
    print("LaTeX rows for tab:comp-algs-others-1 (Dataset 1):")
    print("=" * 70)
    emit(r1)
    print("\nLaTeX rows for tab:comp-algs-others-2 (Dataset 2):")
    print("=" * 70)
    emit(r2)


if __name__ == "__main__":
    main()
