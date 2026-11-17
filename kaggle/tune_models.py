"""Nested hyper-parameter tuning on the fixed folds (same protocol for every model family).

For each outer test fold k, the inner validation fold is v = (k + 1) mod 5. Every configuration in the grid is
trained on the three remaining folds and scored on fold v; the best configuration (ties -> the default) is then
retrained on all four training folds and tested on fold k. The test fold is never used for any choice.

  distilbert: learning rate x epochs (the default 5e-5 / 3 epochs is in the grid)
  sbert:      C of the logistic regression on frozen embeddings (default C = 4 is in the grid)
TF-IDF models are tuned the same way on CPU by scripts/tune_tfidf.py.

Usage (Kaggle): python tune_models.py --family distilbert --datasets djinni,faruque
                python tune_models.py --family sbert --datasets djinni,faruque,stackoverflow
"""
import argparse, copy, itertools, json
import numpy as np, pandas as pd
import common as C

DISTIL_GRID = [{"lr": lr, "epochs": ep} for lr, ep in itertools.product([3e-5, 5e-5], [2, 3, 4])]
DISTIL_DEFAULT = {"lr": 5e-5, "epochs": 3}
SBERT_GRID = [{"C": c} for c in [0.25, 1.0, 4.0, 16.0, 64.0]]
SBERT_DEFAULT = {"C": 4.0}


def nested(df, fit_predict, grid, default, classes, name, tag, extra_cfg):
    fm, rows, choices = [], [], []
    for k in range(5):
        v = (k + 1) % 5
        itr, iva = df[~df.fold.isin([k, v])], df[df.fold == v]
        scores = []
        for cfg in grid:
            P = fit_predict(cfg, itr.text, itr.label, iva.text, classes)
            scores.append(float((np.array(classes)[P.argmax(1)] == iva.label.values).mean()))
            print(f"  {name} outer {k} inner {cfg}: {scores[-1]:.4f}", flush=True)
        best = max(range(len(grid)), key=lambda i: (round(scores[i], 6), grid[i] == default))
        cfg = grid[best]; choices.append({"outer_fold": k, "inner_fold": v, "chosen": cfg,
                                          "inner_scores": [{"cfg": g, "acc": s} for g, s in zip(grid, scores)]})
        tr, te = df[df.fold != k], df[df.fold == k]
        P = fit_predict(cfg, tr.text, tr.label, te.text, classes)
        pred = np.array(classes)[P.argmax(1)]
        fm.append(C.metrics(te.label, pred, P, classes))
        rows.append(pd.DataFrame({"id": te.id, "fold": k, "true": te.label, "pred": pred}))
        print(f"[{tag} | {name}] outer fold {k}: chose {cfg}, test acc {fm[-1]['acc']:.4f}", flush=True)
    p = pd.concat(rows)
    C.save(tag, name, C.summarize(fm), fm, p, {"tuning": choices, "grid": grid, "default": default, "config": extra_cfg})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["distilbert", "sbert"])
    ap.add_argument("--datasets", default="djinni,faruque")
    ap.add_argument("--model", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="quick test: subsample every split")
    a = ap.parse_args()

    if a.family == "distilbert":
        import run_distilbert as D
        base = argparse.Namespace(model=a.model or "distilbert-base-uncased", batch=a.batch, max_len=a.max_len,
                                  seed=a.seed, lr=5e-5, epochs=3)

        def fit_predict(cfg, trt, trl, tet, classes):
            b = copy.copy(base); b.lr, b.epochs = cfg["lr"], cfg["epochs"]; D.seed_all(a.seed)
            if a.limit:
                trt, trl = trt.iloc[:a.limit], trl.iloc[:a.limit]
            return D.train_predict(b, trt, trl, tet, classes)
        grid, default, tag = DISTIL_GRID, DISTIL_DEFAULT, "distilbert_tuned"
    else:
        import run_sbert as S
        from sklearn.linear_model import LogisticRegression
        mname = a.model or "sentence-transformers/all-MiniLM-L6-v2"
        emb = {}

        def fit_predict(cfg, trt, trl, tet, classes):
            X = emb["X"]; m = LogisticRegression(max_iter=4000, class_weight="balanced", C=cfg["C"])
            m.fit(X[trt.index.values], trl.values)
            P = m.predict_proba(X[tet.index.values])
            return P[:, [list(m.classes_).index(c) for c in classes]]
        grid, default, tag = SBERT_GRID, SBERT_DEFAULT, "sbert_" + mname.split("/")[-1] + "_tuned"

    for ds in a.datasets.split(","):
        df = C.load(ds)
        if a.limit:
            df = pd.concat([g.sample(min(len(g), a.limit), random_state=0) for _, g in df.groupby("fold")])
        classes = sorted(df.label.unique())
        if a.family == "sbert":
            full = C.load(ds); Xfull = S.cached(mname, ds, full.text, 128)
            emb["X"] = Xfull                      # df keeps the row index of the full dataset
        nested(df, fit_predict, grid, default, classes, ds, tag, vars(a))
