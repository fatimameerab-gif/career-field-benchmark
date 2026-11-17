"""Nested hyper-parameter tuning for the TF-IDF baselines (CPU), same protocol as kaggle/tune_models.py.

For each outer test fold k the inner validation fold is v = (k + 1) mod 5. Every configuration is trained on the
three remaining folds and scored on fold v; the best one (ties -> the default) is retrained on all four training
folds and tested on fold k. The test fold is never used for any choice.

Grid: n-gram range {(1,1), (1,2)} x C  (SVM: 0.1, 0.3, 1, 3; LR: 1, 4, 16, 64). Defaults: (1,2) and C = 1.

Usage: python scripts/tune_tfidf.py --kinds svm,lr --datasets djinni,faruque,stackoverflow
"""
import argparse, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kaggle"))
import common as C

GRID_C = {"svm": [0.1, 0.3, 1.0, 3.0], "lr": [1.0, 4.0, 16.0, 64.0]}
NGRAMS = [(1, 1), (1, 2)]
DEFAULT = {"ngram": (1, 2), "C": 1.0}


def vec(ngram):
    return TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, max_features=200000)


def clf(kind, c):
    return LogisticRegression(max_iter=4000, class_weight="balanced", C=c) if kind == "lr" else LinearSVC(class_weight="balanced", C=c)


def nested(df, kind, name):
    grid = [{"ngram": g, "C": c} for g in NGRAMS for c in GRID_C[kind]]
    fm, rows, choices = [], [], []
    for k in range(5):
        v = (k + 1) % 5
        itr, iva = df[~df.fold.isin([k, v])], df[df.fold == v]
        scores = []
        for g in NGRAMS:                       # fit each vectoriser once per split
            V = vec(g).fit(itr.text); Xtr, Xva = V.transform(itr.text), V.transform(iva.text)
            for c in GRID_C[kind]:
                m = clf(kind, c).fit(Xtr, itr.label)
                scores.append(((g, c), float((m.predict(Xva) == iva.label.values).mean())))
        cfgs = [{"ngram": g, "C": c} for (g, c), _ in scores]
        best = max(range(len(scores)), key=lambda i: (round(scores[i][1], 6), cfgs[i] == DEFAULT))
        cfg = cfgs[best]
        choices.append({"outer_fold": k, "inner_fold": v, "chosen": {"ngram": list(cfg["ngram"]), "C": cfg["C"]},
                        "inner_scores": [{"ngram": list(g), "C": c, "acc": s} for (g, c), s in scores]})
        tr, te = df[df.fold != k], df[df.fold == k]
        V = vec(cfg["ngram"]).fit(tr.text); m = clf(kind, cfg["C"]).fit(V.transform(tr.text), tr.label)
        Xte = V.transform(te.text); pred = m.predict(Xte)
        proba = m.predict_proba(Xte) if kind == "lr" else None
        fm.append(C.metrics(te.label, pred, proba, list(m.classes_)))
        rows.append(pd.DataFrame({"id": te.id, "fold": k, "true": te.label, "pred": pred}))
        print(f"[tfidf_{kind}_tuned | {name}] outer {k}: chose {cfg}, test acc {fm[-1]['acc']:.4f}", flush=True)
    p = pd.concat(rows)
    C.save(f"tfidf_{kind}_tuned", name, C.summarize(fm), fm, p,
           {"tuning": choices, "grid": [{"ngram": list(g["ngram"]), "C": g["C"]} for g in grid],
            "default": {"ngram": [1, 2], "C": 1.0}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="svm,lr")
    ap.add_argument("--datasets", default="djinni,faruque,stackoverflow")
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        df = C.load(ds)
        for kind in a.kinds.split(","):
            nested(df, kind, ds)
