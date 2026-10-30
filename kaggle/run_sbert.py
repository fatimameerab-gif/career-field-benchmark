"""Sentence-BERT embeddings + Logistic Regression (the original STARRT.ai design), on the fixed folds.

Default models: all-MiniLM-L6-v2 (what STARRT.ai used) and all-mpnet-base-v2 (stronger).
Embeddings are computed once per dataset and cached in results/emb_*.npy.
"""
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import common as C


def embed(model_name, texts, batch):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return m.encode(list(texts), batch_size=batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


def cached(model_name, dataset, texts, batch):
    f = C.out_dir() / f"emb_{model_name.split('/')[-1]}_{dataset}.npy"
    if f.exists():
        return np.load(f)
    e = embed(model_name, texts, batch); np.save(f, e)
    return e


def lr():
    return LogisticRegression(max_iter=4000, class_weight="balanced", C=4.0)


def run(model_name, dataset, folds, batch):
    tag = "sbert_" + model_name.split("/")[-1]
    df = C.load(dataset); X = cached(model_name, dataset, df.text, batch)
    fm, rows = [], []
    for k in folds:
        tr, te = (df.fold != k).values, (df.fold == k).values
        m = lr().fit(X[tr], df.label[tr])
        pred = m.predict(X[te]); proba = m.predict_proba(X[te])
        fm.append(C.metrics(df.label[te], pred, proba, list(m.classes_)))
        rows.append(pd.DataFrame({"id": df.id[te], "fold": k, "true": df.label[te], "pred": pred}))
    p = pd.concat(rows)
    C.save(tag, dataset, C.summarize(fm), fm, p, {"top_confusions": C.top_confusions(p.true, p.pred)})


def forms(model_name, batch):
    tag = "sbert_" + model_name.split("/")[-1]
    tr, fo = C.load("djinni"), C.load_forms()
    Xtr = cached(model_name, "djinni", tr.text, batch)
    m = lr().fit(Xtr, tr.label)
    fo = fo[fo.label.isin(m.classes_)].reset_index(drop=True)
    Xf = embed(model_name, fo.text, batch)
    pred = m.predict(Xf)
    s = C.metrics(fo.label, pred, m.predict_proba(Xf), list(m.classes_))
    C.save(tag, "forms_from_djinni", s, preds=pd.DataFrame({"id": fo.id, "true": fo.label, "pred": pred}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--datasets", default="djinni,stackoverflow,faruque")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    for mn in a.models.split(","):
        for ds in a.datasets.split(","):
            run(mn, ds, folds, a.batch)
        if C.has_forms():
            forms(mn, a.batch)
