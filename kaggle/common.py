"""Shared data loading and metrics for every STARRT.ai experiment.

All models use the same fixed folds (column `fold`, 0-4) created by make_folds.py, so their
numbers are directly comparable. Identical texts never appear in both train and test.
"""
import glob, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATASETS = {"djinni": "djinni_folds.csv", "stackoverflow": "stackoverflow_folds.csv", "faruque": "faruque_folds.csv",
            # Djinni ablations made by scripts/review_analyses.py (same folds, changed text)
            "djinni_nolook": "djinni_nolook_folds.csv", "djinni_masked": "djinni_masked_folds.csv", "djinni_tools": "djinni_tools_folds.csv",
            # perturbed copies of the Djinni folds (scripts/robustness.py)
            "djinni_typo": "djinni_typo_folds.csv", "djinni_shuffle": "djinni_shuffle_folds.csv", "djinni_drop": "djinni_drop_folds.csv"}


def find_data_dir():
    """Works locally (./data) and on Kaggle (/kaggle/input/<any-name>/.../data)."""
    here = Path(__file__).resolve().parent / "data"
    if (here / "label_set.csv").exists():
        return here
    hits = glob.glob("/kaggle/input/**/label_set.csv", recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError("label_set.csv not found. Upload the zip as a Kaggle dataset and attach it.")


def out_dir():
    d = Path("/kaggle/working/results") if os.path.isdir("/kaggle/working") else Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(name):
    df = pd.read_csv(find_data_dir() / DATASETS[name])
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


def has_forms():
    """The Pakistani pilot (forms_test.csv) is not in the public release; every forms step is skipped without it."""
    try:
        return (find_data_dir() / "forms_test.csv").exists()
    except FileNotFoundError:
        return False


def load_forms():
    return pd.read_csv(find_data_dir() / "forms_test.csv")


def label_set():
    return pd.read_csv(find_data_dir() / "label_set.csv")


def metrics(y_true, y_pred, proba=None, classes=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = {"n": int(len(y_true)), "acc": float(accuracy_score(y_true, y_pred)),
         "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    if proba is not None and classes is not None and len(classes) > 3:
        idx = {c: i for i, c in enumerate(classes)}
        top3 = np.argsort(-proba, axis=1)[:, :3]
        m["top3"] = float(np.mean([idx.get(t, -1) in row for t, row in zip(y_true, top3)]))
    return m


def summarize(fold_metrics):
    keys = [k for k in fold_metrics[0] if k != "n"]
    s = {k: round(float(np.mean([f[k] for f in fold_metrics])), 4) for k in keys}
    s.update({k + "_sd": round(float(np.std([f[k] for f in fold_metrics])), 4) for k in keys})
    s["folds"] = len(fold_metrics)
    return s


def top_confusions(y_true, y_pred, k=8):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def save(model, dataset, summary, folds=None, preds=None, extra=None):
    o = out_dir()
    rec = {"model": model, "dataset": dataset, "summary": summary, "folds": folds or [], "time": time.strftime("%Y-%m-%d %H:%M")}
    if extra:
        rec.update(extra)
    with open(o / f"{model}__{dataset}.json", "w") as f:
        json.dump(rec, f, indent=2)
    if preds is not None:
        preds.to_csv(o / f"{model}__{dataset}__predictions.csv", index=False)
    print(f"[{model} | {dataset}]", json.dumps(summary))
