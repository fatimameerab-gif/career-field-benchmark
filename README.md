# Career field prediction benchmark

Code, benchmark data and results for the paper

> **Text Matters More Than the Model: A Leakage-Aware Benchmark for Predicting Computing Career Fields from Text**
> Meerab Fatima, Rohail Iqbal, Ubaid Malik, Shahzeb Khan. FAST-NUCES Peshawar, 2026 (manuscript).

The task is to predict a person's computing career field (13 fields, 4 parent groups, anchored to ESCO and O\*NET) from text they wrote about themselves: a profile, a list of tools, or a list of skills. Every model is trained and tested on the same fixed five folds. In those folds, identical texts never appear in both train and test.

## Main results (5-fold accuracy, mean)

| Model | Djinni (12 cl.) | Faruque (6 cl.) | Stack Overflow (13 cl.) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.879 | 0.914 | 0.450 |
| TF-IDF + linear SVM | 0.886 | 0.923 | 0.438 |
| SBERT all-MiniLM-L6-v2 + LR (original STARRT.ai design) | 0.861 | 0.926 | 0.466 |
| SBERT all-mpnet-base-v2 + LR | 0.873 | 0.921 | 0.463 |
| DistilBERT fine-tuned | **0.889** | 0.931 | **0.516** |
| JobBERT fine-tuned (job-posting BERT) | **0.889** | **0.939** | – |

- DistilBERT is stable across three seeds (Djinni 0.890 ± 0.001, Faruque 0.933 ± 0.002, Stack Overflow 0.517 ± 0.001). Its lead over TF-IDF + SVM on Djinni is small (0.4 points): McNemar p = 0.060 / 0.002 / 0.010 for the three seeds, but the paired bootstrap interval includes zero and the Nadeau–Bengio corrected t-test gives p = 0.32. On Stack Overflow the gain is large and significant (p < 0.001).
- JobBERT (`jjzha/jobbert-base-cased`) matches DistilBERT on Djinni and is best on Faruque (+1.6 points over TF-IDF + SVM, 95% CI +0.7 to +2.6).
- Near-duplicate test rows (TF-IDF cosine ≥ 0.9) do not change which model is best. Masking role words costs TF-IDF + SVM 3 points on Djinni. Reducing the same profiles to their tool lists costs 21 points.
- Zero-shot LLMs on the same fold-0 rows: Qwen2.5-3B-Instruct 0.804 (Djinni) and 0.722 (Faruque); Qwen2.5-7B-Instruct 0.835 and 0.736, against 0.850–0.871 and 0.910–0.943 for the trained models. Without definitions or with a reworded instruction the 3B model drops on Faruque (0.656 / 0.585).
- Perturbed test texts (typos, shuffled word order, 20% words deleted) cost TF-IDF + SVM 0.4–1.3 points and DistilBERT 0.5–1.9 points.
- Two annotators labelled 204 Djinni profiles: Cohen's kappa 0.79 / 0.80 against the dataset labels and 0.73 between them.
- Full tables are in `processed/analysis/`, and every prediction is in `kaggle_results/`.

Repository: https://github.com/fatimameerab-gif/career-field-benchmark

## Repository layout


```
README.md               this file
LICENSE                 MIT licence for the code
DATA_LICENSES.md        licence and attribution for every data file (read before reuse)
LICENSES/               full licence and attribution notices for Djinni (MIT), Stack Overflow (ODbL), Faruque (CC BY), O*NET/ESCO
DATA_STATEMENT.md       data statement: sources, speakers, labels, cleaning, known biases
TECHNICAL_NOTES.md      exact pipeline, hyper-parameters, evaluation protocol, known limitations
CITATION.cff            how to cite
requirements.txt        Python packages
reproduce.sh            rebuild datasets and folds from the raw downloads
raw/README.md           where to download each raw source (raw files are not redistributed)
scripts/
  labels.py             13-class label set and source-to-class mappings
  build_datasets.py     raw/ -> processed/ (cleaning, mapping, de-duplication)
  make_folds.py         processed/ -> kaggle/data/ (per-class cap, grouped 5 folds)
  analyze_results.py    kaggle_results/ -> processed/analysis/ (tables, McNemar, per-class F1)
  make_figures.py       paper figures (optional 3rd argument: bright | classic | muted)
  seed_analysis.py      DistilBERT seeds 42 / 13 / 7 with McNemar tests
  robust_stats.py       macro-F1, Nadeau–Bengio corrected t-test, paired bootstrap CIs, Holm
  robustness.py         perturbed Djinni test texts (typo / shuffle / drop) and the uncapped check
  gpu2_analysis.py      JobBERT, DistilBERT on perturbed texts, LLM prompt variants and Qwen2.5-7B
  label_agreement.py    Cohen's kappa for the two annotators (annotation/)
  review_analyses.py    near-duplicate check, Djinni ablations (role words, tool-only text), LLM tests, Holm correction
  exploratory/          early checks (Faruque duplicate analysis, first TF-IDF baselines)
kaggle/
  data/                 THE BENCHMARK: djinni_folds.csv, stackoverflow_folds.csv, faruque_folds.csv, label_set.csv
  common.py             data loading and metrics shared by all models
  run_tfidf.py, run_sbert.py, run_distilbert.py, run_llm_zeroshot.py
  STARRT_experiments.ipynb   notebook that runs everything on a Kaggle GPU
  STARRT_review_checks.ipynb extra DistilBERT seeds on Djinni and Faruque
  STARRT_gpu_round2.ipynb    Stack Overflow seeds, JobBERT, perturbed texts, LLM prompt variants, Qwen2.5-7B
  README.md             how to run on Kaggle
kaggle_results/         per-model summaries (*.json) and per-row predictions (*__predictions.csv)
processed/analysis/     tables used in the paper
annotation/             label check: 204 Djinni profiles, two annotators' sheets and annotation_labels.csv
figures/                PDF figures of the paper (the manuscript itself is added after publication)
```

## Benchmark files (`kaggle/data/`)

| File | Rows | Columns | Licence |
|---|---|---|---|
| `djinni_folds.csv` | 27,861 | id, source, text, label, fold | MIT (Djinni) |
| `stackoverflow_folds.csv` | 24,380 | id, source, text, label, fold | ODbL 1.0 (Stack Overflow) |
| `faruque_folds.csv` | 2,135 | id, source, text, label (original 6), fold, label13 | CC BY 4.0 (Faruque) |
| `label_set.csv` | 13 | class, parent, O\*NET codes/titles/definition, ESCO labels/codes/definition | see DATA_LICENSES.md |

`fold` is 0–4. To evaluate a new model, train on `fold != k`, test on `fold == k` for k = 0..4, and report the mean. Djinni and Stack Overflow are capped at 3,000 rows per class (random_state = 0) before the folds are made.

The Pakistani pilot set (STARRT.ai Google Form answers) is **not** included, because consent for public release has not been confirmed. The code skips every pilot step when `kaggle/data/forms_test.csv` is missing. Its predictions in `kaggle_results/*forms*` contain only hashed ids and labels, not the texts.

## Reproducing

```bash
pip install -r requirements.txt

# 1. (optional) rebuild the benchmark from the raw downloads: see raw/README.md
bash reproduce.sh            # writes processed/ and kaggle/data/

# 2. CPU baseline (about 15 minutes)
cd kaggle && python run_tfidf.py

# 3. GPU models: run kaggle/STARRT_experiments.ipynb on Kaggle (T4), see kaggle/README.md
#    or locally: python run_sbert.py; python run_distilbert.py; python run_llm_zeroshot.py

# 4. tables and figures from the results
python scripts/analyze_results.py kaggle_results .
python scripts/make_figures.py kaggle_results figures
python scripts/review_analyses.py .     # needs raw/ for the Djinni ablations
python scripts/seed_analysis.py . && python scripts/robust_stats.py .
python scripts/robustness.py .          # needs processed/ (uncapped check)
python scripts/gpu2_analysis.py .
```

`analyze_results.py` and `make_figures.py` read the released predictions, so step 4 runs without a GPU. The pilot table (`forms_33.csv`) and Figure 3 are rebuilt from the released pilot predictions, which contain labels only and no texts.

## Licences in short

- **Code** (`scripts/`, `kaggle/*.py`, notebook): MIT, see `LICENSE`.
- **Data**: each file keeps the licence of its source. Djinni is MIT. Stack Overflow-derived data is ODbL 1.0, so a database you derive from it must stay under ODbL. Faruque et al. is CC BY 4.0, and O\*NET is CC BY 4.0 (modified by us). ESCO is a European Commission classification, used with attribution. Details and attribution text are in `DATA_LICENSES.md`.
- **Models** (downloaded at run time, not included): DistilBERT and the two sentence-transformers models are Apache-2.0. Qwen2.5-3B-Instruct is under the Qwen Research License and Qwen2.5-7B-Instruct under Apache-2.0. The JobBERT model card (`jjzha/jobbert-base-cased`) does not state a licence.

## Citation

See `CITATION.cff`. Please also cite the data sources listed in `DATA_LICENSES.md`.

## Contact

Meerab Fatima (p229407@pwr.nu.edu.pk), FAST-NUCES Peshawar.
