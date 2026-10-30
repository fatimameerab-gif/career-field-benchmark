# Data licences and attribution

Code in this repository is under the MIT licence (`LICENSE`). Data does **not** fall under that licence. Each data file keeps the licence of the source it was derived from, as listed below. If you redistribute any of these files, keep this notice with them.

## Per-file overview

| File(s) | Derived from | Licence of the file | What we changed |
|---|---|---|---|
| `kaggle/data/djinni_folds.csv` | Djinni Recruitment Dataset (candidates, English) | MIT | Joined three text fields; mapped `Primary Keyword` to 12 classes; relabelled ML/AI job titles; removed non-English, short and duplicate texts; capped at 3,000 per class; added folds |
| `kaggle/data/stackoverflow_folds.csv` | Stack Overflow Developer Survey 2023, 2024, 2025 | **ODbL 1.0** (database), DbCL 1.0 (contents) | Filtered respondents; mapped `DevType` to 13 classes; turned five tool columns into one sentence; capped at 3,000 per class; added folds |
| `kaggle/data/faruque_folds.csv` | Student Career Dataset (Faruque, 2024) | CC BY 4.0 | Removed exact duplicates and texts with two labels; added a 13-class mapping (`label13`); added folds |
| `kaggle/data/label_set.csv` | ESCO v1.2.1 and O\*NET 31.0 occupation titles, codes and descriptions | O\*NET parts: CC BY 4.0. ESCO parts: European Commission, reused with attribution | Selected occupations per class; whitespace normalised |
| `kaggle_results/*`, `processed/analysis/*` | our model outputs (ids, true and predicted labels, metrics) | CC BY 4.0 (authors of this repository) | – |
| `paper/sn-jnl.cls`, `paper/sn-mathphys-num.bst` | Springer Nature LaTeX template | Springer Nature template terms | unchanged |

## Required notices

The full notice for each source is also in the `LICENSES/` folder.

**Djinni.** Drushchak, N. and Romanyshyn, M. (2024). *Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings.* Proc. UNLP @ LREC-COLING 2024, 8–13. Released under the MIT licence by lang-uk on Hugging Face (`lang-uk/recruitment-dataset-candidate-profiles-english`). The original MIT copyright and permission notice applies to `djinni_folds.csv`.

**Stack Overflow Developer Survey.** `stackoverflow_folds.csv` contains information from the Stack Overflow Developer Survey 2023–2025 (<https://survey.stackoverflow.co/>). The database is made available under the Open Database License (ODbL) v1.0, <https://opendatacommons.org/licenses/odbl/1-0/>, and any rights in individual contents under the Database Contents License (DbCL) v1.0, <https://opendatacommons.org/licenses/dbcl/1-0/>. `stackoverflow_folds.csv` is a derived database and is therefore also released under ODbL 1.0. If you publicly use a database derived from it, you must release that database under ODbL as well (share-alike) and keep this notice.

**Student Career Dataset.** Faruque, S. H. (2024). *Student Career Dataset*, Mendeley Data, V2, doi:10.17632/4spj4mbpjr.2, licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). We cleaned and relabelled the data; the changes are listed above. Related paper: Faruque, S. H., Khushbu, S. A. and Akter, S. (2024), arXiv:2405.18139.

**O\*NET.** This repository includes information from the O\*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the CC BY 4.0 license. O\*NET® is a trademark of USDOL/ETA. The authors have modified all or some of this information. USDOL/ETA has not approved, endorsed, or tested these modifications.

**ESCO.** This publication uses the ESCO classification of the European Commission (ESCO v1.2.1, <https://esco.ec.europa.eu/>). Only occupation labels, codes and descriptions for 19 occupations are reproduced, in `label_set.csv`. The ESCO download page does not name a specific licence. Reuse follows the terms accepted at download and the European Commission's reuse policy.

## Not included

- **STARRT.ai pilot answers** (46 Google Form answers, Pakistan, Nov 2025). These are not released because consent for public release has not been confirmed. They are available from the authors on reasonable request. Hashed ids and labels for the pilot appear in `kaggle_results/*forms*`, but not the texts.
- **Raw downloads.** See `raw/README.md`.
- **Model weights.** These are downloaded at run time from their original publishers: `distilbert-base-uncased` and `sentence-transformers/all-MiniLM-L6-v2` / `all-mpnet-base-v2` are Apache-2.0, `Qwen/Qwen2.5-3B-Instruct` is under the Qwen Research License, `Qwen/Qwen2.5-7B-Instruct` is Apache-2.0, and the model card of `jjzha/jobbert-base-cased` (Zhang et al., NAACL 2022) does not state a licence.

Licences were checked on 8 October 2026.
