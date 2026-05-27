from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

# ---------------------------------------------------------------------------
# 1. Mixed-distance metric (paper Definition 1)
# ---------------------------------------------------------------------------
# The metric needs to know which columns are categorical and the weight alpha.
# sklearn-extra's CLARA/KMedoids look up metrics by name, so we expose the
# metric as a registered callable and steer it through module-level globals
# that the caller sets via configure_mixed_distance(...).

_CAT_COLS: list[int] = []      # indices of categorical columns
_ALPHA: float = 2.0            # categorical mismatch weight (paper: alpha)


def configure_mixed_distance(cat_cols, alpha):
    """Set the categorical columns and the weight alpha used by mixed_distance."""
    global _CAT_COLS, _ALPHA
    _CAT_COLS = list(cat_cols)
    _ALPHA = float(alpha)


def mixed_distance(X, Y=None, **kwargs):
    """Mixed distance of paper Definition 1.

        d(x,y) = sqrt( ||x_num - y_num||^2  +  alpha * #{categorical mismatches} )

    At alpha = 2 this equals the Euclidean distance on one-hot encoded data
    (paper Theorem 1). Categorical columns are given by the module global
    _CAT_COLS; alpha by _ALPHA. Both are set via configure_mixed_distance().
    """
    if Y is None:
        Y = X
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if _CAT_COLS:
        # count categorical mismatches column by column (memory-efficient:
        # avoids materialising an N x M x m intermediate tensor)
        cat_mismatch = np.zeros((X.shape[0], Y.shape[0]))
        for c in _CAT_COLS:
            cat_mismatch += (X[:, c][:, None] != Y[:, c][None, :])
        Xn = np.delete(X, _CAT_COLS, axis=1)
        Yn = np.delete(Y, _CAT_COLS, axis=1)
    else:
        cat_mismatch = 0.0
        Xn, Yn = X, Y
    num_sq = np.power(euclidean_distances(Xn, Yn), 2)
    return np.sqrt(np.maximum(num_sq + _ALPHA * cat_mismatch, 0.0))


def gower_distance(X, Y=None, **kwargs):
    """Gower's distance: range-normalised Manhattan on numeric features plus the
    overlap (0/1) function on categorical features, averaged over all features.

    Used only as an ablation baseline (paper Section 5, Table `tab:ablation`).
    """
    if Y is None:
        Y = X
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_features = X.shape[1]
    num_cols = [j for j in range(n_features) if j not in _CAT_COLS]
    total = np.zeros((X.shape[0], Y.shape[0]))
    # numeric part: range-normalised |xi - yi|
    for j in num_cols:
        col = np.concatenate([X[:, j], Y[:, j]])
        rng = col.max() - col.min()
        rng = rng if rng > 0 else 1.0
        d = np.abs(X[:, j][:, None] - Y[:, j][None, :]) / rng
        total += d
    # categorical part: overlap
    for j in _CAT_COLS:
        d = (X[:, j][:, None] != Y[:, j][None, :]).astype(float)
        total += d
    return total / n_features


def one_hot_expand(X_train, X_others, cat_cols):
    """Expand categorical columns to one-hot encoding (ablation baseline).

    Returns (X_train_oh, [X_others_oh...]). One-hot Euclidean distance on the
    result is, by paper Theorem 1, equal to the mixed distance at alpha = 2.
    """
    from sklearn.preprocessing import OneHotEncoder
    if not cat_cols:
        return X_train, list(X_others)
    num_cols = [j for j in range(X_train.shape[1]) if j not in cat_cols]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    enc.fit(X_train[:, cat_cols])

    def _exp(A):
        return np.hstack([A[:, num_cols].astype(np.float32),
                          enc.transform(A[:, cat_cols])])

    return _exp(X_train), [_exp(A) for A in X_others]


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_score):
    """Return the six metrics reported in the paper as a dict.

    `y_score` may be a hard 0/1 label or a probability/fraud-ratio in [0,1];
    it is thresholded at 0.5 for the threshold-dependent metrics, exactly as in
    the original notebooks' evaluate_model().
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    try:
        auc = metrics.roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    y_pred = (y_score > 0.5).astype(int)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "mcc": metrics.matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 3. Data loaders (reproduce the notebook preprocessing)
# ---------------------------------------------------------------------------
def _split(df, target_col="Class", seed=42):
    """Stratified 70/10/20 train/val/test split (paper Section 3)."""
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=seed, stratify=y_tr)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def load_dataset1(data_dir=".", seed=42):
    """European Credit Card dataset. All-numerical; cat_cols = [] (paper Sec 5.1).

    Returns dict with X_train/y_train/X_val/y_val/X_test/y_test (numpy arrays),
    feature_names, and cat_cols (empty).
    """
    path = os.path.join(data_dir, "creditcard.csv")
    df = pd.read_csv(path)
    df = df.drop(columns=["Time"])                 # negligible correlation (paper Sec 3)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _split(df, "Class", seed)
    scaler = StandardScaler()                       # only 'Amount' is standardised
    for split in (X_tr,):
        split["Amount"] = scaler.fit_transform(split[["Amount"]])
    for split in (X_val, X_te):
        split["Amount"] = scaler.transform(split[["Amount"]])
    feat = list(X_tr.columns)
    return dict(
        X_train=X_tr.to_numpy(float), y_train=y_tr.to_numpy(int),
        X_val=X_val.to_numpy(float), y_val=y_val.to_numpy(int),
        X_test=X_te.to_numpy(float), y_test=y_te.to_numpy(int),
        feature_names=feat, cat_cols=[], name="Dataset1-EuropeanCC",
    )


def load_dataset2(data_dir=".", seed=42):
    """BankSim dataset. Reproduces notebook-dataset2 preprocessing.

    Final feature order: [age, amount, customer_ID, merchant_ID, category_ID,
    gender_1, gender_2, gender_3]. Categorical columns are customer_ID,
    merchant_ID, category_ID -> indices [2, 3, 4] (paper Sec 5.1 / Table 1).
    """
    path = os.path.join(data_dir, "bs140513_032310.csv")
    df = pd.read_csv(path)

    # integer ID encodings for the high-cardinality categoricals
    for col, new in [("customer", "customer_ID"),
                     ("merchant", "merchant_ID"),
                     ("category", "category_ID")]:
        keys = df.groupby(col).size().reset_index(name="_n")
        keys[new] = range(1, len(keys) + 1)
        df = df.merge(keys[[col, new]], on=col)

    df["age"] = (df["age"].astype(str).str.replace("'", "")
                 .str.replace("U", "7").astype(float))
    df["gender"] = df["gender"].apply(
        lambda x: 1 if x == "'M'" else (2 if x == "'F'" else 3))
    df = pd.get_dummies(df, columns=["gender"])
    df["Class"] = df["fraud"]
    df = df.drop(columns=["customer", "zipcodeOri", "merchant", "zipMerchant",
                          "step", "category", "fraud"])
    # column order matters: categoricals must land at indices 2,3,4
    order = ["age", "amount", "customer_ID", "merchant_ID", "category_ID",
             "gender_1", "gender_2", "gender_3", "Class"]
    df = df[[c for c in order if c in df.columns]]

    X_tr, y_tr, X_val, y_val, X_te, y_te = _split(df, "Class", seed)
    scaler = StandardScaler()                       # standardise age, amount
    for col in ("age", "amount"):
        X_tr[col] = scaler.fit_transform(X_tr[[col]])
        X_val[col] = scaler.transform(X_val[[col]])
        X_te[col] = scaler.transform(X_te[[col]])
    feat = list(X_tr.columns)
    return dict(
        X_train=X_tr.to_numpy(float), y_train=y_tr.to_numpy(int),
        X_val=X_val.to_numpy(float), y_val=y_val.to_numpy(int),
        X_test=X_te.to_numpy(float), y_test=y_te.to_numpy(int),
        feature_names=feat, cat_cols=[2, 3, 4], name="Dataset2-BankSim",
    )


# ---------------------------------------------------------------------------
# 3b. SMOTE (self-contained; avoids an imbalanced-learn version dependency)
# ---------------------------------------------------------------------------
def smote_oversample(X, y, n_synthetic, k=5, seed=42, minority_label=1):
    """Generate `n_synthetic` minority-class points by SMOTE interpolation.

    Standard SMOTE (Chawla et al. 2002): for a random minority point x, pick a
    random one of its k nearest minority neighbours y, and return
    x + lambda * (y - x) with lambda ~ U(0,1). This reproduces the behaviour of
    imblearn.over_sampling.SMOTE used in the original notebooks (all columns,
    including integer-encoded categoricals, are interpolated numerically).
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    rng = np.random.RandomState(seed)
    minority = X[y == minority_label]
    if len(minority) < 2 or n_synthetic < 1:
        return np.empty((0, X.shape[1]))
    kk = min(k, len(minority) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    out = np.empty((n_synthetic, X.shape[1]))
    for s in range(n_synthetic):
        i = rng.randint(len(minority))
        nbr = minority[idx[i, 1 + rng.randint(kk)]]
        out[s] = minority[i] + rng.rand() * (nbr - minority[i])
    return out


# ---------------------------------------------------------------------------
# 4. The INSIGHT pipeline (cluster-aware SMOTE + cluster labelling)
# ---------------------------------------------------------------------------
_PATCH_INSTALLED = False


def _install_metric_patch():
    """Make the named metrics 'mixed' and 'gower' usable by sklearn-extra.

    Newer scikit-learn (>=1.3) strictly validates the `metric` string argument
    of `pairwise_distances` against a fixed whitelist, so the older trick of
    registering a callable in PAIRWISE_DISTANCE_FUNCTIONS no longer works. We
    instead wrap the `pairwise_distances` reference inside sklearn-extra's
    k-medoids module: 'mixed' and 'gower' are routed to our vectorised
    matrix-valued implementations, everything else to the original function.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    import sklearn_extra.cluster._k_medoids as km
    orig_pd = km.pairwise_distances
    orig_pda = km.pairwise_distances_argmin

    def patched_pd(X, Y=None, metric="euclidean", **kw):
        if metric == "mixed":
            return mixed_distance(X, Y)
        if metric == "gower":
            return gower_distance(X, Y)
        return orig_pd(X, Y=Y, metric=metric, **kw)

    def patched_pda(X, Y, *args, metric="euclidean", **kw):
        if metric == "mixed":
            return np.argmin(mixed_distance(X, Y), axis=1)
        if metric == "gower":
            return np.argmin(gower_distance(X, Y), axis=1)
        return orig_pda(X, Y, *args, metric=metric, **kw)

    km.pairwise_distances = patched_pd
    km.pairwise_distances_argmin = patched_pda
    _PATCH_INSTALLED = True


def make_clusterer(algorithm, k, metric, seed=42, n_sampling=250,
                   n_sampling_iter=10):
    """Construct a CLARA / PAM / CLARANS clusterer with the requested metric.

    `metric` is one of: 'euclidean', 'mixed', 'gower'.
    `algorithm` is one of: 'clara', 'pam', 'clarans'.
    CLARANS is approximated by KMedoids(method='alternate'), the closest
    randomised-search variant exposed by sklearn-extra.
    """
    from sklearn_extra.cluster import CLARA, KMedoids
    _install_metric_patch()
    m = metric if metric in ("mixed", "gower") else "euclidean"
    if algorithm == "clara":
        return CLARA(n_clusters=k, random_state=seed, metric=m,
                     n_sampling=n_sampling, n_sampling_iter=n_sampling_iter)
    if algorithm == "pam":
        return KMedoids(n_clusters=k, random_state=seed, metric=m, method="pam")
    if algorithm == "clarans":
        # sklearn-extra has no CLARANS; 'alternate' is its randomised local search
        return KMedoids(n_clusters=k, random_state=seed, metric=m,
                        method="alternate")
    raise ValueError(f"unknown algorithm {algorithm!r}")


def run_insight(data, algorithm="clara", metric="mixed", oversampling="cluster",
                k_range=range(2, 40), os_rate=0.02, alpha=2.0, seed=42,
                select_by="f1"):
    """Train INSIGHT and return (best_k, test_metrics, fit_seconds, best_model).

    Parameters
    ----------
    data         : dict from load_dataset1 / load_dataset2.
    algorithm    : 'clara' | 'pam' | 'clarans'.
    metric       : 'mixed' | 'euclidean' | 'gower'.
    oversampling : 'none' | 'global' | 'cluster'  (cluster = paper Algorithm 5).
    k_range      : cluster counts to sweep on the validation set.
    os_rate      : per-cluster oversampling ratio beta (paper Algorithm 5).
    alpha        : categorical weight for the mixed distance.

    The cluster-aware ('cluster') path reproduces perform_clara_smote() from the
    original notebooks: minority points are SMOTE-generated inside each cluster,
    used to guide re-clustering, then discarded (the scaffolding of Algorithm 5).
    """
    configure_mixed_distance(data["cat_cols"], alpha)
    X_tr, y_tr = data["X_train"], data["y_train"].ravel()
    X_val, y_val = data["X_val"], data["y_val"].ravel()
    X_te, y_te = data["X_test"], data["y_test"].ravel()

    # global SMOTE is applied once, up front, balancing the whole training set
    if oversampling == "global":
        n_syn = int((y_tr == 0).sum() - (y_tr == 1).sum())
        syn = smote_oversample(X_tr, y_tr, n_syn, seed=seed)
        X_fit = np.vstack([X_tr, syn])
        y_fit = np.concatenate([y_tr, np.ones(len(syn), dtype=int)])
    else:
        X_fit, y_fit = X_tr, y_tr

    best = dict(score=-1.0, k=None, test=None, seconds=None, model=None)
    for k in k_range:
        t0 = time.time()
        scaffold_X, scaffold_y = np.array(X_fit), np.array(y_fit)

        if oversampling == "cluster":
            # Phase 1: exploratory clustering, then intra-cluster SMOTE scaffold
            probe = make_clusterer(algorithm, k, metric, seed)
            probe.fit(X_tr)
            lab = probe.predict(X_tr)
            extra_X, extra_y = [], []
            for c in range(k):
                mask = lab == c
                if mask.sum() < 6 or y_tr[mask].sum() < 2:
                    continue
                n_syn = int(mask.sum() * os_rate)
                if n_syn < 1:
                    continue
                syn = smote_oversample(X_tr[mask], y_tr[mask], n_syn, seed=seed)
                if len(syn):
                    extra_X.append(syn)
                    extra_y.append(np.ones(len(syn), dtype=int))
            if extra_X:
                scaffold_X = np.vstack([X_tr] + extra_X)
                scaffold_y = np.concatenate([y_tr] + extra_y)

        # Phase 2: (re-)clustering. Synthetic scaffold guides medoids, then the
        # medoids are frozen and only the *real* training points are labelled.
        model = make_clusterer(algorithm, k, metric, seed)
        model.fit(scaffold_X)
        seconds = time.time() - t0

        # cluster -> fraud-frequency label map (paper Section 3, majority voting)
        lab_tr = model.predict(X_tr)
        freq = {c: (y_tr[lab_tr == c].mean() if (lab_tr == c).any() else 0.0)
                for c in range(k)}

        def _predict(A):
            return np.array([freq[c] for c in model.predict(A)])

        val_m = evaluate(y_val, _predict(X_val))
        if val_m[select_by] > best["score"]:
            best.update(score=val_m[select_by], k=k,
                        test=evaluate(y_te, _predict(X_te)),
                        seconds=seconds, model=model)
    return best["k"], best["test"], best["seconds"], best["model"]


# ---------------------------------------------------------------------------
# 5. Small helpers for LaTeX-ready output
# ---------------------------------------------------------------------------
def fmt(x, nd=5):
    """Format a number for pasting into the manuscript tables."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def latex_row(cells):
    """Join cells into a LaTeX table row."""
    return " & ".join(str(c) for c in cells) + r" \\"


def add_arguments(parser):
    """Attach the arguments common to every experiment script."""
    parser.add_argument("--data-dir", default=".",
                        help="directory holding creditcard.csv and bs140513_032310.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="use a narrow k-range for a fast smoke test")
    return parser
