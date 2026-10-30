"""Robustness and class-cap checks (CPU).

1. Perturbed copies of the Djinni test texts (same rows, same folds), written as fold files so the
   GPU models can be tested on exactly the same perturbations:
     djinni_typo     - 10% of words get one character edit (swap, drop or duplicate)
     djinni_shuffle  - word order shuffled within each sentence
     djinni_drop     - 20% of words removed at random
   TF-IDF + SVM is trained on the clean training folds and tested on each perturbed test fold.
   We report accuracy and the share of test rows whose prediction changed (CheckList-style invariance).
2. Class cap: TF-IDF + SVM on ALL cleaned Djinni and Stack Overflow rows (no 3,000-per-class cap),
   with the same grouped 5-fold protocol, to show how much the cap changes accuracy and macro-F1.

Usage: python3 -I robustness.py <project_dir>
"""
import sys, re, random
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

P = Path(sys.argv[1]); KD = P / "kaggle" / "data"; OUT = P / "processed" / "analysis"


def svm():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
                         LinearSVC(class_weight="balanced"))


def typo(w, r):
    if len(w) < 4 or not w.isalpha(): return w
    i = r.randrange(1, len(w) - 1); op = r.choice(["swap", "drop", "dup"])
    if op == "swap": return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "drop": return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def perturb(text, kind, seed):
    r = random.Random(seed)
    if kind == "typo":
        return " ".join(typo(w, r) if r.random() < 0.10 else w for w in text.split())
    if kind == "shuffle":
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", text):
            ws = sent.split(); r.shuffle(ws); out.append(" ".join(ws))
        return " ".join(out)
    if kind == "drop":
        ws = [w for w in text.split() if r.random() >= 0.20]
        return " ".join(ws) if ws else text
    raise ValueError(kind)


dj = pd.read_csv(KD / "djinni_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
variants = {}
for kind in ["typo", "shuffle", "drop"]:
    v = dj.copy(); v["text"] = [perturb(t, kind, i) for i, t in enumerate(dj.text)]
    v.to_csv(KD / f"djinni_{kind}_folds.csv", index=False); variants[kind] = v

rows = []
clean_pred = pd.Series(index=dj.index, dtype=object); var_pred = {k: pd.Series(index=dj.index, dtype=object) for k in variants}
accs = {k: [] for k in ["clean"] + list(variants)}
for k in range(5):
    tr, te = dj.fold != k, dj.fold == k
    m = svm().fit(dj.text[tr], dj.label[tr])
    clean_pred[te] = m.predict(dj.text[te]); accs["clean"].append((clean_pred[te] == dj.label[te]).mean())
    for kind, v in variants.items():
        var_pred[kind][te] = m.predict(v.text[te]); accs[kind].append((var_pred[kind][te] == dj.label[te]).mean())
rows.append({"test text": "clean", "acc": np.mean(accs["clean"]), "sd": np.std(accs["clean"]), "changed": 0.0,
             "macro_f1": f1_score(dj.label, clean_pred, average="macro")})
for kind in variants:
    rows.append({"test text": kind, "acc": np.mean(accs[kind]), "sd": np.std(accs[kind]),
                 "changed": float((var_pred[kind] != clean_pred).mean()), "macro_f1": f1_score(dj.label, var_pred[kind], average="macro")})
rob = pd.DataFrame(rows); rob.to_csv(OUT / "robust_perturbation_tfidf.csv", index=False)
print("Robustness, TF-IDF + SVM on Djinni:\n", rob.round(4).to_string(index=False))

# 2. uncapped data
cap = []
for name, f in [("djinni", "djinni_clean.csv"), ("stackoverflow", "stackoverflow_2023_2025.csv")]:
    d = pd.read_csv(P / "processed" / f).dropna(subset=["text", "label"]).reset_index(drop=True)
    g = pd.factorize(d.text.str.lower())[0]; d["fold"] = -1
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d.text, d.label, g)):
        d.loc[te, "fold"] = k
    pred = pd.Series(index=d.index, dtype=object); a = []
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        pred[te] = svm().fit(d.text[tr], d.label[tr]).predict(d.text[te]); a.append((pred[te] == d.label[te]).mean())
    cap.append({"dataset": name, "rows": len(d), "acc": np.mean(a), "sd": np.std(a), "macro_f1": f1_score(d.label, pred, average="macro"),
                "majority": d.label.value_counts(normalize=True).iloc[0]})
cap = pd.DataFrame(cap); cap.to_csv(OUT / "robust_uncapped_tfidf.csv", index=False)
print("\nUncapped, TF-IDF + SVM:\n", cap.round(4).to_string(index=False))
