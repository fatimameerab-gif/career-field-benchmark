"""Hierarchical TF-IDF + LR classifiers on the fixed folds (CPU), compared with the flat TF-IDF + LR model.

two-stage: a group classifier picks one of the four parent groups, then a field classifier trained only on that
           group's training rows picks the field (a wrong group cannot be undone).
soft:      every field is scored by P(group) x P(field | group) and the best-scoring field is the prediction.
Both use the default TF-IDF + LR settings of kaggle/run_tfidf.py. Paired comparison with the flat model's
out-of-fold predictions (kaggle_results/tfidf_lr__<dataset>__predictions.csv) by the exact McNemar test.

Usage: python scripts/hierarchical_tfidf.py --datasets djinni,stackoverflow
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
import common as C


def lr():
    return LogisticRegression(max_iter=4000, class_weight="balanced")


def run(ds):
    df = C.load(ds)
    parent = C.label_set().set_index("class")["parent"].to_dict()
    df["group"] = df.label.map(parent)
    rows = []
    for k in range(5):
        tr, te = df[df.fold != k], df[df.fold == k]
        V = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000).fit(tr.text)
        Xtr, Xte = V.transform(tr.text), V.transform(te.text)
        g = lr().fit(Xtr, tr.group); Pg = g.predict_proba(Xte); groups = list(g.classes_)
        fields, score = [], []
        sub = {}
        for gi, gname in enumerate(groups):
            m = tr.group.values == gname
            labs = sorted(tr.label[m].unique())
            sub[gname] = (lr().fit(Xtr[m], tr.label[m]) if len(labs) > 1 else None, labs)
        soft_cols, soft_names = [], []
        for gi, gname in enumerate(groups):
            mdl, labs = sub[gname]
            P = mdl.predict_proba(Xte) if mdl is not None else np.ones((Xte.shape[0], 1))
            names = list(mdl.classes_) if mdl is not None else labs
            soft_cols.append(P * Pg[:, [gi]]); soft_names += names
        S = np.hstack(soft_cols)
        soft = np.array(soft_names)[S.argmax(1)]
        gpred = np.array(groups)[Pg.argmax(1)]
        two = np.empty(len(te), dtype=object)
        for gname in groups:
            m = gpred == gname
            if not m.any(): continue
            mdl, labs = sub[gname]
            two[m] = mdl.predict(Xte[m]) if mdl is not None else labs[0]
        rows.append(pd.DataFrame({"id": te.id.values, "fold": k, "true": te.label.values, "two_stage": two, "soft": soft}))
        print(f"[{ds}] fold {k}: two-stage {np.mean(two == te.label.values):.4f}  soft {np.mean(soft == te.label.values):.4f}", flush=True)
    p = pd.concat(rows)
    flat = pd.read_csv(ROOT / "kaggle_results" / f"tfidf_lr__{ds}__predictions.csv")[["id", "fold", "pred"]].rename(columns={"pred": "flat"})
    p = p.merge(flat, on=["id", "fold"])
    out = []
    for v in ["two_stage", "soft"]:
        a, b = (p[v] == p.true).values, (p.flat == p.true).values
        b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
        out.append({"dataset": ds, "variant": v, "n": len(p), "acc": a.mean(), "flat_acc": b.mean(),
                    "only_variant": b10, "only_flat": b01, "mcnemar_p": pv})
    o = ROOT / "processed" / "analysis"; o.mkdir(parents=True, exist_ok=True)
    p.to_csv(o / f"hierarchical_tfidf__{ds}__predictions.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="djinni,stackoverflow"); a = ap.parse_args()
    res = []
    for ds in a.datasets.split(","):
        res += run(ds)
    r = pd.DataFrame(res); print(r.to_string(index=False))
    r.to_csv(ROOT / "processed" / "analysis" / "hierarchical_tfidf.csv", index=False)
