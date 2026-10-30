"""DistilBERT seed robustness: seeds 42 (main run), 13 and 7 on Djinni, Faruque and Stack Overflow.

Usage: python3 -I seed_analysis.py <project_dir>
Writes processed/analysis/review_seeds.csv and review_seeds_mcnemar.csv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

P = Path(sys.argv[1]); R = P / "kaggle_results"; OUT = P / "processed" / "analysis"
SEEDS = {42: "distilbert", 13: "distilbert_seed13", 7: "distilbert_seed7"}
OTHERS = {"tfidf_svm": "TF-IDF + SVM", "tfidf_lr": "TF-IDF + LR", "sbert_all-mpnet-base-v2": "mpnet + LR", "sbert_all-MiniLM-L6-v2": "MiniLM + LR"}
rows, tests = [], []
for ds in ["djinni", "faruque", "stackoverflow"]:
    accs, f1s = [], []
    for seed, tag in SEEDS.items():
        j = json.load(open(R / f"{tag}__{ds}.json"))
        a = float(np.mean([f["acc"] for f in j["folds"]])); f = float(np.mean([f["macro_f1"] for f in j["folds"]]))
        accs.append(a); f1s.append(f)
        rows.append({"dataset": ds, "seed": seed, "acc": a, "acc_fold_sd": float(np.std([f["acc"] for f in j["folds"]])), "macro_f1": f})
        p = pd.read_csv(R / f"{tag}__{ds}__predictions.csv")
        for m, name in OTHERS.items():
            o = pd.read_csv(R / f"{m}__{ds}__predictions.csv")
            assert (o.id.values == p.id.values).all() and (o.true.values == p.true.values).all()
            a1 = (p.true == p.pred).values; b1 = (o.true == o.pred).values
            n01 = int((a1 & ~b1).sum()); n10 = int((~a1 & b1).sum())
            tests.append({"dataset": ds, "seed": seed, "vs": name, "distil_only": n01, "other_only": n10,
                          "p": binomtest(n01, n01 + n10, 0.5).pvalue})
    rows.append({"dataset": ds, "seed": "mean of 3", "acc": float(np.mean(accs)), "acc_fold_sd": float(np.std(accs, ddof=1)), "macro_f1": float(np.mean(f1s))})
s = pd.DataFrame(rows); s.to_csv(OUT / "review_seeds.csv", index=False)
t = pd.DataFrame(tests); t.to_csv(OUT / "review_seeds_mcnemar.csv", index=False)
print(s.round(4).to_string(index=False)); print(t.round(4).to_string(index=False))
