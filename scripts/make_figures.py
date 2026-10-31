"""Paper figures. File names follow the figure numbers printed in the paper:

  Figure_1.pdf  5-fold accuracy of the supervised models on the three benchmarks      (Fig. 1)
  Figure_2.pdf  zero-shot Qwen2.5-3B vs trained models on the same fold-0 rows         (Fig. 2)
  Figure_3.pdf  accuracy on the 33 Pakistani pilot answers, 95% Wilson intervals        (Fig. 3)
  Figure_4.pdf  DistilBERT confusion matrix on Djinni (row %)                           (Fig. 4)
  Figure_5.pdf  per-class F1 on Djinni for three models                                 (Fig. 5)

Every number is read from kaggle_results/ and processed/analysis/ (nothing is typed by hand).
Bars start at zero; each model keeps one colour in every figure (colour-blind checked palette).

Usage: python3 make_figures.py <results_dir> <out_dir> [bright|classic|muted]
   e.g. python3 scripts/make_figures.py kaggle_results paper/Figures
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
ANALYSIS = R.parent / "processed" / "analysis"

plt.rcParams.update({
    # Computer Modern (cmr10), the same font family as the LaTeX paper; no LaTeX install needed
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.axisbelow": True,
    "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
INK = "#222222"

# one colour per model, used in every figure. Three palettes; choose with the 3rd argument.
#   classic: greys for TF-IDF, blues for sentence embeddings, navy for DistilBERT, dark red for the LLM
#   muted:   Paul Tol's colour-blind-safe "muted" scheme
#   bright:  saturated colour-blind-checked scheme
PALETTES = {
    "classic": ["#d9d9d9", "#969696", "#b5cde6", "#5b8fc7", "#1f3d7a", "#a50f15"],
    "muted":   ["#88ccee", "#44aa99", "#ddcc77", "#999933", "#332288", "#cc6677"],
    "bright":  ["#eb6834", "#1baf7a", "#eda100", "#008300", "#2a78d6", "#e87ba4"],
}
PALETTE = sys.argv[3] if len(sys.argv) > 3 else "bright"
EDGE = ("white", 0.8) if PALETTE == "bright" else ("#333333", 0.4)   # thin dark outline keeps light bars visible in print
_c = PALETTES[PALETTE]
#        key in results                 legend label
MODELS = [("tfidf_lr",                 "TF-IDF + LR",            _c[0]),
          ("tfidf_svm",                "TF-IDF + SVM",           _c[1]),
          ("sbert_all-MiniLM-L6-v2",   "MiniLM + LR*",           _c[2]),
          ("sbert_all-mpnet-base-v2",  "mpnet + LR",             _c[3]),
          ("distilbert",               "DistilBERT fine-tuned",  _c[4]),
          ("llm_qwen2.5-3b_zeroshot",  "Qwen2.5-3B zero-shot",   _c[5])]
# Qwen2.5-7B (Figure 2 only): darker shade of the LLM colour, so both LLMs read as one family
LLM7 = ("llm_qwen2.5-7b_zeroshot", "Qwen2.5-7B zero-shot", {"bright": "#9c2f6b", "classic": "#67000d", "muted": "#882255"}[sys.argv[3] if len(sys.argv) > 3 else "bright"])
COLOR = {k: c for k, _, c in MODELS + [LLM7]}
LABEL = {k: n for k, n, _ in MODELS + [LLM7]}
# names used inside processed/analysis/*.csv
ANAME = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "SBERT MiniLM + LR (original STARRT.ai)",
         "sbert_all-mpnet-base-v2": "SBERT mpnet + LR", "distilbert": "DistilBERT fine-tuned", "llm_qwen2.5-3b_zeroshot": "Qwen2.5-3B zero-shot"}
SUPERVISED = [m for m, _, _ in MODELS[:5]]
SHORT = {"AI/ML Engineering": "AI/ML", "Cybersecurity": "Security", "Data Engineering & Databases": "Data Eng.",
         "Data Science & Analytics": "Data Sci.", "DevOps & Cloud": "DevOps", "Game Development": "Game",
         "Mobile Development": "Mobile", "Networking & SysAdmin": "Network", "Product & Project Management": "PM",
         "QA & Testing": "QA", "UI/UX Design": "UI/UX", "Web Development": "Web", "Embedded & IoT": "Embedded"}
DJ_ORDER = ["Web Development", "Mobile Development", "Game Development", "Data Science & Analytics",
            "Data Engineering & Databases", "AI/ML Engineering", "DevOps & Cloud", "Cybersecurity",
            "Networking & SysAdmin", "QA & Testing", "UI/UX Design", "Product & Project Management"]


def summ(m, ds):
    """mean and SD recomputed from the per-fold values, so labels round exactly like the paper's tables"""
    f = [x["acc"] for x in json.load(open(R / f"{m}__{ds}.json"))["folds"]]
    return {"acc": float(np.mean(f)), "acc_sd": float(np.std(f))}


def preds(m, ds):
    return pd.read_csv(R / f"{m}__{ds}__predictions.csv")


def bars(ax, models, vals, sds=None):
    x = np.arange(len(models))
    b = ax.bar(x, vals, 0.78, color=[COLOR[m] for m in models], edgecolor=EDGE[0], linewidth=EDGE[1],
               yerr=sds, capsize=2 if sds is not None else 0, error_kw={"lw": 0.7, "ecolor": INK})
    for xi, v, s in zip(x, vals, sds if sds is not None else [0] * len(vals)):
        ax.text(xi, v + s + 0.015, f"{round(v + 1e-12, 3):.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks([]); ax.set_ylim(0, 1.08); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True)
    return b


def legend_below(fig, models, ncol, y=0.0):
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[m], edgecolor=EDGE[0], linewidth=EDGE[1]) for m in models]
    fig.legend(h, [LABEL[m] for m in models], loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=7)


# ---------- Figure 1: 5-fold accuracy of the supervised models ----------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, (ds, title) in zip(axes, [("djinni", "(a) Djinni, 12 classes"), ("faruque", "(b) Faruque, 6 classes"),
                                  ("stackoverflow", "(c) Stack Overflow, 13 classes")]):
    s = [summ(m, ds) for m in SUPERVISED]
    bars(ax, SUPERVISED, [x["acc"] for x in s], [x["acc_sd"] for x in s])
    ax.set_title(title, fontsize=8)
axes[0].set_ylabel(r"Accuracy (mean $\pm$ SD, 5 folds)")
fig.tight_layout(rect=(0, 0.1, 1, 1)); legend_below(fig, SUPERVISED, 5)
fig.savefig(OUT / "Figure_1.pdf"); plt.close(fig)

# ---------- Figure 2: zero-shot LLMs vs trained models on the same rows (samples of all five folds) ----------
same = pd.read_csv(ANALYSIS / "gpu3_llm_allfolds.csv")      # written by scripts/gpu3_analysis.py
ANAME2 = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "MiniLM + LR",
          "sbert_all-mpnet-base-v2": "mpnet + LR", "distilbert": "DistilBERT", "llm_qwen2.5-3b_zeroshot": "Qwen2.5-3B",
          "llm_qwen2.5-7b_zeroshot": "Qwen2.5-7B"}
ALL2 = [m for m, _, _ in MODELS] + [LLM7[0]]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, ds in zip(axes, ["djinni", "faruque"]):
    sub = same[same.dataset == ds]
    vals = [sub[sub.model == ANAME2[m]].acc.iloc[0] for m in ALL2]
    lo = [sub[sub.model == ANAME2[m]].ci_low.iloc[0] for m in ALL2]; hi = [sub[sub.model == ANAME2[m]].ci_high.iloc[0] for m in ALL2]
    x = np.arange(len(ALL2))
    ax.bar(x, vals, 0.78, color=[COLOR[m] for m in ALL2], edgecolor=EDGE[0], linewidth=EDGE[1],
           yerr=[np.array(vals) - lo, np.array(hi) - vals], capsize=2, error_kw={"lw": 0.7, "ecolor": INK})
    for xi, v, h in zip(x, vals, hi):
        ax.text(xi, h + 0.015, f"{round(v + 1e-12, 3):.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks([]); ax.set_ylim(0, 1.08); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.yaxis.grid(True)
    n = int(sub.n.iloc[0])
    ax.set_title(f"({'a' if ds == 'djinni' else 'b'}) {'Djinni' if ds == 'djinni' else 'Faruque'}, samples of all 5 folds (n = {n:,})", fontsize=8)
axes[0].set_ylabel("Accuracy (95% Wilson interval)")
fig.tight_layout(rect=(0, 0.14, 1, 1)); legend_below(fig, ALL2, 4)
fig.savefig(OUT / "Figure_2.pdf"); plt.close(fig)

# ---------- Figure 3: Pakistani pilot, correct answers with 95% Wilson intervals ----------
forms = pd.read_csv(ANALYSIS / "forms_33.csv").set_index("model")
ALL = [m for m, _, _ in MODELS]


def wilson(k, n, z=1.96):
    p_ = k / n; d = 1 + z * z / n; c = (p_ + z * z / (2 * n)) / d
    h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(4.8, 2.4))
for i, m in enumerate(ALL):
    r = forms.loc[ANAME[m]]; lo_, hi_ = wilson(r.correct, r.n)
    ax.plot([lo_, hi_], [i, i], color=COLOR[m] if PALETTE == "bright" else "#444444", lw=1.6 if PALETTE == "bright" else 1.0, solid_capstyle="round")
    if PALETTE != "bright":   # forest-plot style: dark interval line with end ticks, coloured dot
        ax.plot([lo_, lo_], [i - 0.15, i + 0.15], color="#444444", lw=1.0); ax.plot([hi_, hi_], [i - 0.15, i + 0.15], color="#444444", lw=1.0)
    ax.plot(r.acc, i, "o", ms=7, zorder=3, color=COLOR[m], markeredgecolor="white" if PALETTE == "bright" else "#333333", markeredgewidth=1.2 if PALETTE == "bright" else 0.6)
    ax.text(hi_ + 0.015, i, f"{int(r.correct)}/{int(r.n)}", va="center", fontsize=6.5, color=INK)
maj = 6 / 33
ax.axvline(maj, ls="--", lw=0.7, color="#777777"); ax.text(maj + 0.01, len(ALL) - 0.45, "majority class (6/33)", fontsize=6, color="#555555")
ax.set_yticks(range(len(ALL))); ax.set_yticklabels([LABEL[m] for m in ALL], fontsize=7)
ax.set_xlim(0, 1.1); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_ylim(len(ALL) - 0.3, -0.7)
ax.xaxis.grid(True); ax.set_xlabel("Accuracy on 33 Pakistani answers (95% Wilson interval)")
fig.tight_layout(); fig.savefig(OUT / "Figure_3.pdf"); plt.close(fig)

# ---------- Figure 4: DistilBERT confusion matrix on Djinni (row %) ----------
p = preds("distilbert", "djinni")
labs = [c for c in DJ_ORDER if c in set(p.true)]
cm = confusion_matrix(p.true, p.pred, labels=labs); rowp = cm / cm.sum(1, keepdims=True) * 100
shown = np.where(rowp >= 1, rowp, np.nan)                     # cells under 1% stay white
fig, ax = plt.subplots(figsize=(4.6, 4.0))
# pcolormesh keeps every cell as a vector shape, so the matrix stays sharp at any zoom
ax.pcolormesh(np.arange(len(labs) + 1) - 0.5, np.arange(len(labs) + 1) - 0.5, np.ma.masked_invalid(shown),
              cmap="Blues", vmin=0, vmax=100, edgecolors="none", rasterized=False)
ax.set_xlim(-0.5, len(labs) - 0.5); ax.set_ylim(len(labs) - 0.5, -0.5); ax.set_aspect("equal")
for i in range(len(labs)):
    for j in range(len(labs)):
        if rowp[i, j] >= 1:
            ax.text(j, i, f"{rowp[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if rowp[i, j] > 55 else INK)
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
ax.set_xticklabels([SHORT[c] for c in labs], rotation=60, ha="right", fontsize=6.5); ax.set_yticklabels([SHORT[c] for c in labs], fontsize=6.5)
for b in [2.5, 5.5, 8.5]:
    ax.axhline(b, color="#444444", lw=0.6); ax.axvline(b, color="#444444", lw=0.6)
ax.set_xlabel("Predicted field"); ax.set_ylabel("True field")
for s in ax.spines.values(): s.set_visible(True)
fig.tight_layout(); fig.savefig(OUT / "Figure_4.pdf"); plt.close(fig)

# ---------- Figure 5: per-class F1 on Djinni, three models ----------
THREE = ["tfidf_svm", "sbert_all-MiniLM-L6-v2", "distilbert"]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
w = 0.27
for k, m in enumerate(THREE):
    q = preds(m, "djinni"); f = f1_score(q.true, q.pred, labels=labs, average=None)
    ax.bar(np.arange(len(labs)) + (k - 1) * w, f, w, color=COLOR[m], edgecolor=EDGE[0], linewidth=EDGE[1], label=LABEL[m])
ax.set_xticks(range(len(labs))); ax.set_xticklabels([SHORT[c] for c in labs], fontsize=7)
ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.yaxis.grid(True)
ax.set_ylabel("F1 (Djinni, 5 folds pooled)")
for b in [2.5, 5.5, 8.5]:
    ax.axvline(b, color="#999999", lw=0.5, ls=":")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(OUT / "Figure_5.pdf"); plt.close(fig)
print("figures written to", OUT)
