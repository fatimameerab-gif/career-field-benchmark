"""Label agreement check: two annotators labelled the same 204 Djinni benchmark profiles (17 per class,
drawn with random_state=7) without seeing the dataset labels. Computes Cohen's kappa between the
annotators and between each annotator and the dataset label, with bootstrap 95% CIs (2,000 resamples).

Usage: python3 -I label_agreement.py <project_dir>
Writes annotation/annotation_labels.csv (id, dataset label, A, B) and processed/analysis/label_agreement.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score as ck

P = Path(sys.argv[1]); A = P / "annotation"; OUT = P / "processed" / "analysis"
a = pd.read_excel(A / "Annotator_A_label_sheet.xlsx").iloc[:, [0, 2]]; a.columns = ["item", "A"]
b = pd.read_excel(A / "Annotator_B_label_sheet.xlsx").iloc[:, [0, 2]]; b.columns = ["item", "B"]
k = pd.read_csv(A / "sample_ids_and_dataset_labels.csv")
d = k.merge(a, on="item").merge(b, on="item"); assert len(d) == 204 and d[["A", "B"]].notna().all().all()
d[["item", "id", "label", "A", "B"]].to_csv(A / "annotation_labels.csv", index=False)
rng = np.random.default_rng(0)


def row(name, x, y, sub):
    s = sub.reset_index(drop=True); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(s), len(s)); bs.append(ck(s[x].values[i], s[y].values[i]))
    return {"pair": name, "n": len(s), "kappa": ck(s[x], s[y]), "ci_low": np.percentile(bs, 2.5),
            "ci_high": np.percentile(bs, 97.5), "raw_agreement": (s[x] == s[y]).mean()}


rows = [row("A vs B", "A", "B", d), row("A vs dataset", "A", "label", d), row("B vs dataset", "B", "label", d)]
dec = d[(d.A != "Unclear") & (d.B != "Unclear")]
rows += [row("A vs B, both decided", "A", "B", dec), row("A vs dataset, A decided", "A", "label", d[d.A != "Unclear"]),
         row("B vs dataset, B decided", "B", "label", d[d.B != "Unclear"])]
r = pd.DataFrame(rows)
agree = d[(d.A == d.B) & (d.A != "Unclear")]
r = pd.concat([r, pd.DataFrame([{"pair": "dataset label = annotators' shared label", "n": len(agree),
                                 "raw_agreement": (agree.A == agree.label).mean()}])])
r.to_csv(OUT / "label_agreement.csv", index=False)
print(f"Unclear: A {int((d.A == 'Unclear').sum())}, B {int((d.B == 'Unclear').sum())}")
print(r.round(3).to_string(index=False))
print("\nDataset label matches the shared annotator label, by class:")
print(agree.assign(ok=agree.A == agree.label).groupby("label").ok.agg(["mean", "size"]).round(2).to_string())
