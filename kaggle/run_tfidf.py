"""TF-IDF + Logistic Regression / linear SVM baselines on the fixed folds (CPU, a few minutes)."""
import argparse
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import common as C


def make(kind):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced") if kind == "lr" else LinearSVC(class_weight="balanced")
    return make_pipeline(vec, clf)


def run(dataset, kind, folds):
    df = C.load(dataset); fm, rows = [], []
    for k in folds:
        tr, te = df[df.fold != k], df[df.fold == k]
        m = make(kind).fit(tr.text, tr.label)
        pred = m.predict(te.text)
        proba = m.predict_proba(te.text) if kind == "lr" else None
        fm.append(C.metrics(te.label, pred, proba, list(m.classes_)))
        rows.append(pd.DataFrame({"id": te.id, "fold": k, "true": te.label, "pred": pred}))
    p = pd.concat(rows)
    C.save(f"tfidf_{kind}", dataset, C.summarize(fm), fm, p, {"top_confusions": C.top_confusions(p.true, p.pred)})


def forms(kind):
    tr, fo = C.load("djinni"), C.load_forms()
    m = make(kind).fit(tr.text, tr.label)
    fo = fo[fo.label.isin(m.classes_)]
    pred = m.predict(fo.text)
    proba = m.predict_proba(fo.text) if kind == "lr" else None
    s = C.metrics(fo.label, pred, proba, list(m.classes_))
    C.save(f"tfidf_{kind}", "forms_from_djinni", s, preds=pd.DataFrame({"id": fo.id, "true": fo.label, "pred": pred}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="djinni,stackoverflow,faruque")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    for ds in a.datasets.split(","):
        for kind in ["lr", "svm"]:
            run(ds, kind, folds)
    for kind in ["lr", "svm"] if C.has_forms() else []:
        forms(kind)
