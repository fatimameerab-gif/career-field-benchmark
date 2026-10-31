# Technical notes

These notes give the exact settings behind every number in the paper. Where the notes and the code ever disagree, the code is correct.

## 1. Pipeline

```
raw/ ──build_datasets.py──> processed/*.csv ──make_folds.py──> kaggle/data/*_folds.csv
                                                                     │
                     run_tfidf / run_sbert / run_distilbert / run_llm_zeroshot (kaggle/)
                                                                     │
                                                       kaggle_results/*.json, *__predictions.csv
                                                                     │
                                   analyze_results.py ──> processed/analysis/*.csv
                                   make_figures.py    ──> paper/Figures/Figure_1-5.pdf
```

## 2. Label set (`scripts/labels.py`)

- **Classes.** There are 13 fields in 4 parent groups (listed in `DATA_STATEMENT.md`). Each field has ESCO v1.2.1 occupation labels and, where one exists, O\*NET-SOC codes. O\*NET has no separate AI/ML engineering occupation. `build_datasets.py` copies the official titles and descriptions into `label_set.csv`.
- **Djinni** (`DJINNI`).
  - JavaScript, PHP, Node.js and Ruby → Web.
  - iOS, Android and Flutter → Mobile.
  - Unity → Game.
  - Data Science and Data Analyst → Data Science & Analytics.
  - Data Engineer and SQL → Data Engineering.
  - DevOps → DevOps & Cloud.
  - Security → Cybersecurity.
  - Sysadmin → Networking & SysAdmin.
  - QA and QA Automation → QA & Testing.
  - Design → UI/UX.
  - Project Manager, Product Manager and Scrum Master → Product & Project Management.
  - Data Science or Python profiles whose title matches `(?i)\b(machine learning|deep learning|ml|ai|computer vision|cv engineer|nlp|llm|mlops)\b` → AI/ML Engineering.
  - Design profiles whose title matches graphic, illustration, motion, 3D, brand, print, interior or fashion are removed.
  - All other keywords are dropped.
- **Stack Overflow.** `SO_DEVTYPE` is used for 2025 and `SO_DEVTYPE_2023_2024` for the two earlier years. Role names changed between years, which is why there are two maps. Generic roles (for example "Developer, desktop or enterprise applications", students, researchers, executives) are not mapped.
- **Faruque.** The original 6 labels are used for the benchmark. `label13` maps 4 of them to our fields. "Development" and "Software Development and Engineering" have no single field, so their `label13` is empty.

## 3. Cleaning (`scripts/build_datasets.py`)

**All sources**
- Whitespace is normalised and bullets are removed.
- Ids are deterministic: the source id, or the md5 of the lower-cased text for sources without ids.

**Djinni**
- Text = `Moreinfo` + `Highlights` + `Looking For`. `Position` (the job title) and `CV` are never in the text.
- A row is kept if it has less than 10% non-ASCII characters and at least 15 words.
- Exact duplicate texts are removed.
- 210,250 → 128,891 rows.

**Stack Overflow**
- `MainBranch` must be one of the three working-developer answers.
- The text has the form `Languages: a, b. Databases: c. Platforms: d. Frameworks: e. Collaboration tools: f.`. Only these five fields are used because they were asked in all three years, so the format does not reveal the survey year.
- At least 2 fields must be non-empty.
- Result: 114,879 rows.

**Faruque**
- 4,076 rows → 2,135 after removing 1,900 exact duplicates and 41 texts that carry two different labels.
- `scripts/exploratory/eval_faruque.py` compares cross-validation with and without the duplicates.

**Pilot**
- 46 answers → 44 after removing empty and duplicate answers. 34 map to our fields and 33 fall in the 12 Djinni classes.

## 4. Folds (`scripts/make_folds.py`)

- Djinni and Stack Overflow are capped at 3,000 rows per class with `groupby(label).sample(random_state=0)`. Faruque is not capped.
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by the lower-cased text, so identical texts are always in the same fold.
- The fold files are the benchmark. Every model reads the same files, which makes paired tests possible.

## 5. Models (`kaggle/`)

| Model | Settings |
|---|---|
| TF-IDF + LR | word 1–2-grams, `sublinear_tf`, `min_df=2`, `max_features=200,000`. `LogisticRegression(max_iter=4000, class_weight="balanced")` |
| TF-IDF + SVM | same features. `LinearSVC(class_weight="balanced")` |
| SBERT + LR | `all-MiniLM-L6-v2` (original STARRT.ai design) or `all-mpnet-base-v2`. Normalised embeddings, default max sequence length. `LogisticRegression(C=4, max_iter=4000, class_weight="balanced")` |
| DistilBERT | `distilbert-base-uncased`, 3 epochs, AdamW lr 5e-5, weight decay 0.01, batch 32, max length 256, linear schedule with 6% warm-up, gradient clip 1.0, fp16, seed 42 for every fold. No validation split and no early stopping, so the test fold is never used for tuning. |
| JobBERT | `jjzha/jobbert-base-cased` (BERT-base further pre-trained on ~3.2M job-posting sentences, Zhang et al. 2022), same recipe as DistilBERT, seed 42, Djinni and Faruque only |
| Qwen2.5-3B zero-shot | `Qwen/Qwen2.5-3B-Instruct`, fp16, no training. Details in the list below. |
| Qwen2.5-7B zero-shot | `Qwen/Qwen2.5-7B-Instruct`, fp16, same prompt, spread over two T4 GPUs (`--device_map auto`) |

How the zero-shot model works:
- The prompt lists every field with the first sentence (at most 22 words) of its ESCO definition, or of its O\*NET definition if there is no ESCO one. The six Faruque labels use short hand-written definitions (`FARUQUE_DEF`).
- The person's text is cut to 250 words.
- Each class name followed by `<|im_end|>` is scored by its summed log-probability, and the highest score wins. The output is therefore always a valid label.
- The model is run on a stratified sample of each test fold: at most 40 rows per class, `random_state=0` (`--folds`). Fold 0 gives 480 rows for Djinni and 212 for Faruque; all five folds give 2,400 and 1,057.
- On the pilot, all 13 classes are allowed.
- Prompt variants for the 3B model (`--prompt_variant`): `nodef` lists field names only; `plain` keeps the definitions but uses a reworded, job-matching instruction.

Hyper-parameters were set once, before the results were seen, and never tuned on test folds.

Compute:
- TF-IDF ran on CPU.
- All other models ran on one Kaggle T4 GPU: about 1.5 h for DistilBERT, 3 datasets × 5 folds, and about 30 min for the LLM.

## 6. Metrics and statistics (`kaggle/common.py`, `scripts/analyze_results.py`)

- Accuracy and macro-F1 are computed per fold. Tables report the mean ± SD (population SD over 5 folds).
- Top-3 accuracy is reported for models that give probabilities.
- **Paired tests:** exact McNemar (two-sided binomial on discordant pairs) on the pooled out-of-fold predictions, which share the same rows in the same order (checked by asserts).
- **LLM comparison:** the trained models' fold-0 predictions are taken on exactly the LLM's sampled rows. Faruque ids are not unique, so rows are matched by position (checked by asserts).
- **Pilot:** models are trained on all of Djinni and tested on 33 answers. Accuracy is given with 95% Wilson intervals.
- **Hierarchical accuracy:** a prediction counts as correct if it is in the same parent group as the true label.

## 7. Reproducibility checks done

- `reproduce.sh` run from the raw downloads gave fold files identical to the released ones (same rows, labels and folds), with and without the private pilot file.
- `analyze_results.py` on the released predictions gives exactly the numbers in the paper.
- GPU results can vary slightly between GPU types and library versions (non-deterministic CUDA kernels), so a re-run of DistilBERT may differ in the third decimal. TF-IDF and SBERT + LR results are deterministic given the same embeddings.

## 8. Known limitations

- Labels are self-reported or platform categories, not expert-annotated. Only Djinni was checked by human annotators (204 profiles, two annotators from the author team; see section 11).
- Djinni has no Embedded & IoT class, so the Djinni models predict 12 classes.
- The pilot has only 33 answers, so its Wilson intervals are wide (for example 0.727–0.952 for 29/33).
- The LLMs were tested on stratified samples of the five test folds, not on every row, because of GPU time. The prompt variants were tested on fold 0 only.
- DistilBERT used three seeds on all three datasets; JobBERT used three seeds and was not run on Stack Overflow. The other models are deterministic given the folds.

## 9. Leakage and validity checks (`scripts/review_analyses.py`)

- **Near-duplicates.** For each test row, the highest TF-IDF cosine similarity to the training rows of the same fold. At a threshold of 0.9:
  - Djinni: 0 rows
  - Faruque: 273 rows (12.8%)
  - Stack Overflow: 1,044 rows (4.3%)

  Re-scoring every model without these rows cost at most 0.011 on Faruque and 0.008 on Stack Overflow. The best models did not change: JobBERT (seed 42: 0.939 → 0.932; `gpu2_near_duplicates.csv`) and DistilBERT remained the best models on Djinni and Faruque, and DistilBERT on Stack Overflow.
- **Djinni ablations** (TF-IDF + SVM, same folds):

  | Text | Accuracy |
  |---|---|
  | Full | 0.886 |
  | Without "Looking For" | 0.882 |
  | Role and keyword words masked | 0.854 |
  | Both | 0.852 |

- **Controlled text type.** Djinni profiles were reduced to the Stack Overflow tools they mention, keeping only profiles with at least 2 tools (12,133). On these same profiles, TF-IDF + SVM scored 0.885 on the full text and 0.676 on the tool list alone.
- **Zero-shot LLM.** Wilson intervals and exact McNemar tests against every trained model on the same rows, with Holm adjustment.
- **Holm correction.** Applied to all 12 DistilBERT McNemar tests.
- **Extra seeds.** DistilBERT was re-run on Djinni and Faruque with seeds 13 and 7 (`scripts/seed_analysis.py`, `kaggle/STARRT_review_checks.ipynb`).

  | Dataset | Seed 42 | Seed 13 | Seed 7 | Mean ± SD | McNemar p vs TF-IDF + SVM (42 / 13 / 7) |
  |---|---|---|---|---|---|
  | Djinni | 0.889 | 0.890 | 0.890 | 0.890 ± 0.001 | 0.060 / 0.002 / 0.010 |
  | Faruque | 0.931 | 0.932 | 0.935 | 0.933 ± 0.002 | 0.13 / 0.062 / 0.017 |

  The DistilBERT versions of the Djinni ablations (`djinni_nolook`, `djinni_masked`, `djinni_tools`) can be run with the same notebook. They were not run for the paper, which reports these ablations with TF-IDF + SVM only.

## 10. Second GPU round (`kaggle/STARRT_gpu_round2.ipynb`, `scripts/gpu2_analysis.py`)

- **Stack Overflow seeds.** DistilBERT seeds 42 / 13 / 7: 0.516 / 0.516 / 0.518 (mean 0.517 ± 0.001); McNemar p < 0.001 against TF-IDF + SVM for every seed. Holm over all nine seed tests leaves the Djinni and Faruque values unchanged.
- **JobBERT.** Djinni 0.889 ± 0.003 (macro-F1 0.870), Faruque 0.939 ± 0.013 (0.947). Against TF-IDF + SVM: Djinni +0.004 (bootstrap 95% CI +0.001 to +0.007, McNemar p = 0.021, Nadeau–Bengio p = 0.25); Faruque +0.016 (CI +0.007 to +0.026, Nadeau–Bengio p = 0.007). Against DistilBERT on Djinni: McNemar p = 0.60.
- **DistilBERT on perturbed texts** (same trained models as the clean run):

  | Test text | Accuracy | Macro-F1 | Predictions changed |
  |---|---|---|---|
  | Clean | 0.889 | 0.870 | – |
  | Typos | 0.884 | 0.864 | 3.0% |
  | Shuffled | 0.870 | 0.850 | 6.5% |
  | 20% dropped | 0.877 | 0.858 | 5.3% |

- **LLM variants** (fold-0 sample, Wilson 95% CI):

  | Model / prompt | Djinni (480) | Faruque (212) |
  |---|---|---|
  | Qwen2.5-3B, definitions (default) | 0.804 [0.766, 0.837] | 0.722 [0.658, 0.778] |
  | Qwen2.5-3B, names only | 0.783 [0.744, 0.818] | 0.656 [0.589, 0.716] |
  | Qwen2.5-3B, reworded instruction | 0.800 [0.762, 0.833] | 0.585 [0.518, 0.649] |
  | Qwen2.5-7B, definitions | 0.835 [0.800, 0.866] | 0.736 [0.673, 0.791] |

  On Djinni, Qwen2.5-7B is 1.5–3.6 points below every trained model, but no difference is significant after Holm (raw p 0.053–0.48). On Faruque every trained model is better (p < 0.001).

## 11. Label agreement (`annotation/`, `scripts/label_agreement.py`)

- 204 Djinni benchmark profiles, 17 per class, drawn with `random_state=7` from all five folds.
- Two annotators from the author team labelled them independently in a spreadsheet with a dropdown of the 12 fields plus "Unclear", using the field definitions, without seeing the dataset labels or each other's labels.
- Results (Cohen's kappa, bootstrap 95% CI):

  | Pair | n | kappa | 95% CI | Raw agreement |
  |---|---|---|---|---|
  | A vs B | 204 | 0.734 | 0.669–0.797 | 0.755 |
  | A vs dataset | 204 | 0.787 | 0.726–0.845 | 0.804 |
  | B vs dataset | 204 | 0.798 | 0.738–0.850 | 0.814 |
  | A vs dataset, "Unclear" removed | 192 | 0.841 | 0.783–0.891 | 0.854 |
  | B vs dataset, "Unclear" removed | 193 | 0.847 | 0.791–0.898 | 0.860 |

- Where both annotators chose the same field (149 profiles), the dataset label matched in 95.3% of cases. The 7 mismatches are in AI/ML Engineering, Data Science & Analytics, DevOps & Cloud and Game Development.
- `annotation/annotation_labels.csv` has the id, the dataset label and both annotators' labels.

## 12. Third GPU round (`kaggle/STARRT_gpu_round3.ipynb`, `scripts/gpu3_analysis.py`)

- **JobBERT seeds 42 / 13 / 7.** Djinni 0.889 / 0.889 / 0.890 (0.889 ± 0.001); Faruque 0.939 / 0.934 / 0.930 (0.934 ± 0.005). Against TF-IDF + SVM, Nadeau–Bengio p: Djinni 0.25 / 0.19 / 0.23, Faruque 0.007 / 0.12 / 0.12 (Holm over the six: only Faruque seed 42 < 0.05).
- **LLMs on samples of all five folds** (40 per class per fold; trained models scored on the same rows with their out-of-fold predictions):

  | Model | Djinni (2,400) | Faruque (1,057) |
  |---|---|---|
  | Qwen2.5-3B | 0.779 [0.762, 0.795] | 0.733 [0.706, 0.759] |
  | Qwen2.5-7B | 0.838 [0.823, 0.853] | 0.743 [0.715, 0.768] |
  | Trained models | 0.854–0.872 | 0.923–0.940 |

  Per-fold accuracy, Djinni: 3B 0.758–0.804, 7B 0.829–0.856. 7B vs 3B: Djinni p < 0.001, Faruque p = 0.52. 7B vs trained models on Djinni: Holm-adjusted p ≤ 0.018 except MiniLM + LR (0.070).
