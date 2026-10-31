"""Analyses for the third GPU round (kaggle/STARRT_gpu_round3.ipynb):

1. JobBERT with seeds 42 / 13 / 7 on Djinni and Faruque: accuracy and macro-F1 per seed, mean and SD over
   seeds, and for every seed against TF-IDF + SVM: exact McNemar, Nadeau-Bengio corrected resampled t-test
   and paired bootstrap 95% CI; McNemar against DistilBERT with the same seed. Holm over the six
   JobBERT-vs-SVM McNemar tests and, separately, the six Nadeau-Bengio tests.
2. Zero-shot LLMs on stratified samples (40 per class) of ALL five test folds (fold 0 from earlier rounds,
   folds 1-4 from this round): pooled accuracy with Wilson CI, mean and SD over folds, and exact McNemar
   tests of each LLM against every trained model on exactly the same rows (trained models' out-of-fold
   predictions, matched by position within each fold), and 7B against 3B. Holm over all LLM tests.

Usage: python3 -I gpu3_analysis.py <project_dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.metrics import f1_score

P = Path(sys.argv[1]); R = P / "kaggle_results"; OUT = P / "processed" / "analysis"; KD = P / "kaggle" / "data"
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p_ = k / n; d = 1 + z * z / n; c = (p_ + z * z / (2 * n)) / d
    h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b):
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def folds(tag, ds):
    return json.load(open(R / f"{tag}__{ds}.json"))["folds"]


def preds(tag, ds):
    return pd.read_csv(R / f"{tag}__{ds}__predictions.csv")


# ------------------------------------------------------------ 1. JobBERT seeds
JB = {42: "jobbert", 13: "jobbert_seed13", 7: "jobbert_seed7"}
DB = {42: "distilbert", 13: "distilbert_seed13", 7: "distilbert_seed7"}
rows = []
for ds in ["djinni", "faruque"]:
    svm = preds("tfidf_svm", ds); csvm = (svm.true == svm.pred).values
    fsvm = np.array([f["acc"] for f in folds("tfidf_svm", ds)])
    for seed, tag in JB.items():
        p = preds(tag, ds); assert (p.id.values == svm.id.values).all() and (p.true.values == svm.true.values).all()
        c = (p.true == p.pred).values; fj = folds(tag, ds)
        n01, n10, pm = mcnemar(c, csvm)
        d = np.array([f["acc"] for f in fj]) - fsvm
        t = d.mean() / np.sqrt((1 / len(d) + 0.25) * d.var(ddof=1)); pnb = 2 * stats.t.sf(abs(t), len(d) - 1)
        diff = c.astype(float) - csvm.astype(float)
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        db = preds(DB[seed], ds); assert (db.id.values == p.id.values).all()
        _, _, pdb = mcnemar(c, (db.true == db.pred).values)
        rows.append({"dataset": ds, "seed": seed, "acc": np.mean([f["acc"] for f in fj]), "acc_fold_sd": np.std([f["acc"] for f in fj]),
                     "macro_f1": np.mean([f["macro_f1"] for f in fj]), "diff_vs_svm": diff.mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "mcnemar_p_vs_svm": pm, "nb_p_vs_svm": pnb, "mcnemar_p_vs_distilbert_same_seed": pdb})
jb = pd.DataFrame(rows)
jb["mcnemar_p_holm"] = holm(jb.mcnemar_p_vs_svm.values); jb["nb_p_holm"] = holm(jb.nb_p_vs_svm.values)
jb.to_csv(OUT / "gpu3_jobbert_seeds.csv", index=False)
summ = jb.groupby("dataset").agg(acc_mean=("acc", "mean"), acc_sd=("acc", lambda x: np.std(x, ddof=1)),
                                 f1_mean=("macro_f1", "mean")).reset_index()
summ.to_csv(OUT / "gpu3_jobbert_seed_summary.csv", index=False)

# ------------------------------------------------------------ 2. LLMs on all five folds
LLM = {"llm_qwen2.5-3b_zeroshot": "Qwen2.5-3B", "llm_qwen2.5-7b_zeroshot": "Qwen2.5-7B"}
TRAINED = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "MiniLM + LR",
           "sbert_all-mpnet-base-v2": "mpnet + LR", "distilbert": "DistilBERT", "jobbert": "JobBERT"}
acc_rows, test_rows = [], []
for ds in ["djinni", "faruque"]:
    full = pd.read_csv(KD / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    corr = {n: [] for n in list(LLM.values()) + list(TRAINED.values())}; per_fold = {n: [] for n in LLM.values()}
    true_all, pred_all = [], {n: [] for n in LLM.values()}
    for k in range(5):
        tk = full[full.fold == k]
        samp = pd.concat([g.sample(min(len(g), 40), random_state=0) for _, g in tk.groupby("label")])
        pos = [list(tk.index).index(i) for i in samp.index]
        for tag, name in LLM.items():
            L = preds(tag, f"{ds}_fold{k}_sample")
            assert (L.id.values == samp.id.values).all() and (L.true.values == samp.label.values).all(), (tag, ds, k)
            c = (L.true == L.pred).values; corr[name].append(c); per_fold[name].append(c.mean()); pred_all[name].append(L.pred.values)
        true_all.append(samp.label.values)
        for tag, name in TRAINED.items():
            o = preds(tag, ds); o = o[o.fold == k].reset_index(drop=True).iloc[pos]
            assert (o.true.values == samp.label.values).all(), (tag, ds, k)
            corr[name].append((o.true == o.pred).values)
    corr = {n: np.concatenate(v) for n, v in corr.items()}; true_all = np.concatenate(true_all)
    for name, c in corr.items():
        lo, hi = wilson(c.sum(), len(c))
        rec = {"dataset": ds, "model": name, "n": len(c), "acc": c.mean(), "ci_low": lo, "ci_high": hi}
        if name in LLM.values():
            rec.update({"fold_mean": np.mean(per_fold[name]), "fold_sd": np.std(per_fold[name]),
                        "fold0": per_fold[name][0], "macro_f1": f1_score(true_all, np.concatenate(pred_all[name]), average="macro")})
        acc_rows.append(rec)
    for llm in LLM.values():
        for name in TRAINED.values():
            n01, n10, p = mcnemar(corr[llm], corr[name])
            test_rows.append({"dataset": ds, "llm": llm, "vs": name, "acc_llm": corr[llm].mean(), "acc_other": corr[name].mean(),
                              "llm_only_right": n01, "other_only_right": n10, "p": p})
    n01, n10, p = mcnemar(corr["Qwen2.5-7B"], corr["Qwen2.5-3B"])
    test_rows.append({"dataset": ds, "llm": "Qwen2.5-7B", "vs": "Qwen2.5-3B", "acc_llm": corr["Qwen2.5-7B"].mean(),
                      "acc_other": corr["Qwen2.5-3B"].mean(), "llm_only_right": n01, "other_only_right": n10, "p": p})
acc = pd.DataFrame(acc_rows); acc.to_csv(OUT / "gpu3_llm_allfolds.csv", index=False)
tt = pd.DataFrame(test_rows); tt["p_holm"] = holm(tt.p.values); tt.to_csv(OUT / "gpu3_llm_allfolds_tests.csv", index=False)

pd.set_option("display.width", 250)
for n, df in [("JobBERT seeds", jb), ("JobBERT seed summary", summ), ("LLM, all five folds", acc), ("LLM tests", tt)]:
    print(f"\n== {n}\n", df.round(4).to_string(index=False))
