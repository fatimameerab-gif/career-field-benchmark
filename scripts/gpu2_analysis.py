"""Analyses for the second GPU round (kaggle/STARRT_gpu_round2.ipynb):

1. JobBERT (jjzha/jobbert-base-cased) fine-tuned on Djinni and Faruque, same recipe as DistilBERT:
   accuracy / macro-F1, exact McNemar vs DistilBERT (seed 42) and TF-IDF + SVM, Nadeau-Bengio test and
   paired bootstrap CI vs TF-IDF + SVM.
2. DistilBERT tested on the perturbed Djinni test texts (typo / shuffle / drop): accuracy, macro-F1 and the
   share of test rows whose prediction changed relative to the clean text (same trained models).
3. Zero-shot LLM variants on the same fold-0 samples: Qwen2.5-3B with three prompts (default = with ESCO /
   O*NET definitions, nodef = class names only, plain = definitions kept, reworded job-matching instruction) and
   Qwen2.5-7B with the default prompt. Wilson CIs, and exact McNemar tests of Qwen2.5-7B against
   Qwen2.5-3B and every trained model on the same rows, with Holm adjustment.

Usage: python3 -I gpu2_analysis.py <project_dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.metrics import f1_score

P = Path(sys.argv[1]); R = P / "kaggle_results"; OUT = P / "processed" / "analysis"
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


# ---------------------------------------------------------------- 1. JobBERT
rows = []
for ds in ["djinni", "faruque"]:
    j = preds("jobbert", ds); fj = folds("jobbert", ds)
    rec = {"dataset": ds, "acc": np.mean([f["acc"] for f in fj]), "acc_sd": np.std([f["acc"] for f in fj]),
           "macro_f1": np.mean([f["macro_f1"] for f in fj]), "macro_f1_sd": np.std([f["macro_f1"] for f in fj])}
    cj = (j.true == j.pred).values
    for tag, name in [("distilbert", "DistilBERT"), ("tfidf_svm", "SVM")]:
        o = preds(tag, ds)
        assert (o.id.values == j.id.values).all() and (o.true.values == j.true.values).all()
        n01, n10, p = mcnemar(cj, (o.true == o.pred).values)
        rec[f"vs_{name}_jobbert_only"] = n01; rec[f"vs_{name}_other_only"] = n10; rec[f"vs_{name}_p"] = p
    o = preds("tfidf_svm", ds)
    d = np.array([f["acc"] for f in fj]) - np.array([f["acc"] for f in folds("tfidf_svm", ds)])
    t = d.mean() / np.sqrt((1 / len(d) + 0.25) * d.var(ddof=1))
    rec["vs_SVM_nb_p"] = 2 * stats.t.sf(abs(t), len(d) - 1)
    diff = cj.astype(float) - (o.true == o.pred).values.astype(float)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    rec["vs_SVM_diff"] = diff.mean(); rec["vs_SVM_ci_low"] = np.percentile(boot, 2.5); rec["vs_SVM_ci_high"] = np.percentile(boot, 97.5)
    rows.append(rec)
jb = pd.DataFrame(rows); jb.to_csv(OUT / "gpu2_jobbert.csv", index=False)

# near-duplicate check for JobBERT (same procedure as review_analyses.py: highest TF-IDF cosine of each test
# text to the training texts of its fold; rows with similarity >= 0.9 removed)
from sklearn.feature_extraction.text import TfidfVectorizer
nd = []
for ds in ["djinni", "faruque"]:
    df = pd.read_csv(P / "kaggle" / "data" / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(df.text).transform(df.text)
    ms = np.zeros(len(df))
    for k in range(5):
        te = np.where(df.fold == k)[0]; Xtr = X[np.where(df.fold != k)[0]].T.tocsc()
        for s0 in range(0, len(te), 2000):
            blk = te[s0:s0 + 2000]; ms[blk] = (X[blk] @ Xtr).max(axis=1).toarray().ravel()
    df["maxsim"] = ms
    pool = pd.concat([df[df.fold == k] for k in range(5)]).reset_index(drop=True)
    keep = (pool.maxsim < 0.9).values
    rec = {"dataset": ds, "near_dup_rows": int((~keep).sum())}
    for tag, name in [("jobbert", "JobBERT"), ("distilbert", "DistilBERT"), ("tfidf_svm", "TF-IDF + SVM"),
                      ("sbert_all-MiniLM-L6-v2", "MiniLM + LR"), ("sbert_all-mpnet-base-v2", "mpnet + LR"), ("tfidf_lr", "TF-IDF + LR")]:
        pr = preds(tag, ds)
        assert (pr.true.values == pool.label.values).all() and (pr.id.values == pool.id.values).all()
        ok = (pr.true == pr.pred).values
        rec[f"{name} | all"] = ok.mean(); rec[f"{name} | clean"] = ok[keep].mean()
    nd.append(rec)
nd = pd.DataFrame(nd); nd.to_csv(OUT / "gpu2_near_duplicates.csv", index=False)
print("\n== Near-duplicates incl. JobBERT\n", nd.round(4).T.to_string())

# ---------------------------------------------------------------- 2. DistilBERT robustness
clean = preds("distilbert_robust", "djinni")
rows = [{"test text": "clean", "acc": np.mean([f["acc"] for f in folds("distilbert_robust", "djinni")]),
         "sd": np.std([f["acc"] for f in folds("distilbert_robust", "djinni")]), "changed": 0.0,
         "macro_f1": f1_score(clean.true, clean.pred, average="macro")}]
for kind in ["typo", "shuffle", "drop"]:
    v = preds("distilbert_robust", f"djinni_{kind}")
    assert (v.id.values == clean.id.values).all() and (v.true.values == clean.true.values).all()
    fv = folds("distilbert_robust", f"djinni_{kind}")
    rows.append({"test text": kind, "acc": np.mean([f["acc"] for f in fv]), "sd": np.std([f["acc"] for f in fv]),
                 "changed": float((v.pred != clean.pred).mean()), "macro_f1": f1_score(v.true, v.pred, average="macro")})
rob = pd.DataFrame(rows); rob.to_csv(OUT / "gpu2_perturbation_distilbert.csv", index=False)

# ---------------------------------------------------------------- 3. LLM variants
LLM = {"llm_qwen2.5-3b_zeroshot": "Qwen2.5-3B, definitions", "llm_qwen2.5-3b_nodef": "Qwen2.5-3B, names only",
       "llm_qwen2.5-3b_plain": "Qwen2.5-3B, reworded instruction", "llm_qwen2.5-7b_zeroshot": "Qwen2.5-7B, definitions"}
TRAINED = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "MiniLM + LR",
           "sbert_all-mpnet-base-v2": "mpnet + LR", "distilbert": "DistilBERT", "jobbert": "JobBERT"}
var_rows, test_rows = [], []
for ds in ["djinni", "faruque"]:
    base = preds("llm_qwen2.5-3b_zeroshot", f"{ds}_fold0_sample")
    corr = {}
    for tag, name in LLM.items():
        p = preds(tag, f"{ds}_fold0_sample")
        assert (p.id.values == base.id.values).all() and (p.true.values == base.true.values).all()
        c = (p.true == p.pred).values; corr[name] = c
        lo, hi = wilson(c.sum(), len(c))
        var_rows.append({"dataset": ds, "model": name, "n": len(c), "acc": c.mean(), "ci_low": lo, "ci_high": hi,
                         "macro_f1": f1_score(p.true, p.pred, average="macro")})
    # trained models: fold-0 predictions on the same rows. Faruque ids are not unique, so rows are matched
    # by position inside fold 0, exactly as in review_analyses.py.
    full = pd.read_csv(P / "kaggle" / "data" / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    t0 = full[full.fold == 0]
    samp = pd.concat([g.sample(min(len(g), 40), random_state=0) for _, g in t0.groupby("label")])
    assert (samp.id.values == base.id.values).all()
    pos = [list(t0.index).index(i) for i in samp.index]
    for tag, name in TRAINED.items():
        o = preds(tag, ds); o = o[o.fold == 0].reset_index(drop=True).iloc[pos]
        assert (o.true.values == base.true.values).all()
        corr[name] = (o.true == o.pred).values
    seven = corr["Qwen2.5-7B, definitions"]
    for name, c in corr.items():
        if name == "Qwen2.5-7B, definitions": continue
        n01, n10, p = mcnemar(seven, c)
        test_rows.append({"dataset": ds, "vs": name, "acc_other": c.mean(), "acc_7b": seven.mean(),
                          "7b_only_right": n01, "other_only_right": n10, "p": p})
    # 3B prompt variants against the default 3B prompt
    for name in ["Qwen2.5-3B, names only", "Qwen2.5-3B, reworded instruction"]:
        n01, n10, p = mcnemar(corr[name], corr["Qwen2.5-3B, definitions"])
        test_rows.append({"dataset": ds, "vs": f"{name} vs 3B definitions", "acc_other": corr[name].mean(),
                          "acc_7b": np.nan, "7b_only_right": n01, "other_only_right": n10, "p": p})
var = pd.DataFrame(var_rows); var.to_csv(OUT / "gpu2_llm_variants.csv", index=False)
tt = pd.DataFrame(test_rows); tt["p_holm"] = holm(tt.p.values); tt.to_csv(OUT / "gpu2_llm_tests.csv", index=False)

pd.set_option("display.width", 220)
for name, df in [("JobBERT", jb), ("DistilBERT robustness", rob), ("LLM variants", var), ("LLM tests", tt)]:
    print(f"\n== {name}\n", df.round(4).to_string(index=False))
