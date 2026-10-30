"""Extra statistics: macro-F1 for every model, Nadeau-Bengio corrected resampled t-test on fold
accuracies, paired bootstrap 95% CI for accuracy differences, and Holm correction for the seed tests.

Usage: python3 -I robust_stats.py <project_dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P = Path(sys.argv[1]); R = P / "kaggle_results"; OUT = P / "processed" / "analysis"
M = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "MiniLM + LR",
     "sbert_all-mpnet-base-v2": "mpnet + LR", "distilbert": "DistilBERT"}
DS = ["djinni", "faruque", "stackoverflow"]
rng = np.random.default_rng(0)


def folds(m, ds):
    return json.load(open(R / f"{m}__{ds}.json"))["folds"]


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


# 1. macro-F1 for every model (mean +- population SD over folds)
f1 = [{"model": n, "dataset": ds, "macro_f1": np.mean([f["macro_f1"] for f in folds(m, ds)]),
       "macro_f1_sd": np.std([f["macro_f1"] for f in folds(m, ds)])} for m, n in M.items() for ds in DS]
pd.DataFrame(f1).to_csv(OUT / "robust_macro_f1.csv", index=False)

# 2. Nadeau-Bengio corrected resampled t-test (k=5 folds, n_test/n_train = 1/4) and 3. paired bootstrap CI
rows = []
for ds in DS:
    b = pd.read_csv(R / f"distilbert__{ds}__predictions.csv")
    for m, n in M.items():
        if m == "distilbert": continue
        d = np.array([a["acc"] for a in folds("distilbert", ds)]) - np.array([a["acc"] for a in folds(m, ds)])
        k = len(d); var = d.var(ddof=1)
        t = d.mean() / np.sqrt((1 / k + 0.25) * var) if var > 0 else np.inf
        p = 2 * stats.t.sf(abs(t), k - 1)
        o = pd.read_csv(R / f"{m}__{ds}__predictions.csv")
        assert (o.id.values == b.id.values).all()
        diff = (b.true == b.pred).values.astype(float) - (o.true == o.pred).values.astype(float)
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        rows.append({"dataset": ds, "vs": n, "diff": diff.mean(), "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5), "nb_t": t, "nb_p": p})
nb = pd.DataFrame(rows); nb["nb_p_holm"] = holm(nb.nb_p.values)
nb.to_csv(OUT / "robust_nb_bootstrap.csv", index=False)

# 4. Holm over the nine seed tests (DistilBERT vs TF-IDF + SVM, 3 datasets x 3 seeds)
s = pd.read_csv(OUT / "review_seeds_mcnemar.csv"); s = s[s.vs == "TF-IDF + SVM"].copy()
s["p_holm"] = holm(s.p.values); s.to_csv(OUT / "robust_seed_holm.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(f1).round(3).to_string(index=False)); print(nb.round(4).to_string(index=False)); print(s.round(4).to_string(index=False))
