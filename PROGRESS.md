<div align="center">

# 📍 ChargeIQ — Progress Board

**Live status of the 23CSE301 Machine Learning capstone, tracked against the official rubric.**

<br>

![Overall](https://img.shields.io/badge/Deliverables%20in%20repo-21%2F50%20marks-FFB300?style=for-the-badge)
![Review 1](https://img.shields.io/badge/Review%201-19%2F19%20implementation-2E7D32?style=for-the-badge)
![Review 2](https://img.shields.io/badge/Review%202-2%2F25-9E9E9E?style=for-the-badge)

![Regression](https://img.shields.io/badge/🔋%20Regression-10%2F10%20algorithms-2E7D32?style=flat-square)
![Classification A](https://img.shields.io/badge/🚗%20Classification%20Part%20A-5%2F5-2E7D32?style=flat-square)
![Classification B](https://img.shields.io/badge/🚗%20Classification%20Part%20B-0%2F5-9E9E9E?style=flat-square)
![Clustering](https://img.shields.io/badge/🔌%20Clustering-0%2F2-9E9E9E?style=flat-square)
![Bonus GUI](https://img.shields.io/badge/🎁%20Bonus%20GUI-Not%20started-9E9E9E?style=flat-square)

*Last updated: 21 September 2026 · 51 commits · at [`571fbbd`](https://github.com/Roots26Hz/ChargeIQ/commit/571fbbd)*

</div>

> **Legend** — ✅ done · 🚧 in progress · ⬜ not started · ⚠️ at risk
>
> **Badge meaning** — marks whose deliverable is physically in the repository. Review 1: **19/19** implementation marks done; the remaining 6 (presentation + viva) are earned on the day. Review 2: **2/25** — README (C2) and `requirements.txt` (C3).

---

## 🗓️ Scope & Deadlines

| Review | When | Scope | Marks |
|:--|:--|:--|--:|
| **Review 1** | Just before mid-semester examinations | Full **Regression** track + **Classification Part A** | 25 |
| **Review 2** | Towards the end of the semester | **Classification Part B** + full **Clustering** track | 25 |
| **Bonus** | With Review 2 | GUI / deployed application | +2 |

Both reviews are in-person presentations. **Every team member must be able to explain any part of the submitted work.**

---

## 🧱 Milestone Timeline

### 🔋 Regression Track — `notebooks/regression.ipynb`

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 00 | Repository initialised & structure scaffolded | `8c734fe` … `ca3fd24` | ✅ |
| 01 | Dataset links uploaded | `0376391` | ✅ |
| 02 | Regression track — setup & config | `5407507` | ✅ |
| 03 | Structural join — one row per discharge cycle | `e10adaa` | ✅ |
| 04 | Feature extraction from raw per-cycle CSVs | `693db4e` | ✅ |
| 05 | Exploratory Data Analysis | `befa20d` | ✅ |
| 06 | Data cleaning | `f23fa58` | ✅ |
| 07 | 🎯 Target engineering — **SOH** | `b5c627a` | ✅ |
| 08 | Feature set finalised | `e85eb2b` | ✅ |
| 09 | Train/test split (grouped by battery) & scaling | `9caacdd` | ✅ |
| 10 | Feature engineering — `cycle_index_squared` | `17db5d6` | ✅ |
| 11 | Algorithms 1–5 — Linear · Ridge · Lasso · ElasticNet · Polynomial | `41938ef` … `46c20ba` | ✅ |
| 12 | Algorithm 6 — Decision Tree Regressor | `a732cd0` | ✅ |
| 13 | Algorithm 7 — Random Forest Regressor | `05f4035` | ✅ |
| 14 | Algorithm 8 — Gradient Boosting Regressor | `9b2650e` | ✅ |
| 15 | Algorithm 9 — Support Vector Regressor | `23971bf` | ✅ |
| 16 | Algorithm 10 — K-Nearest Neighbors Regressor | `21e2d59` | ✅ |
| 17 | Comparative evaluation — all 10 algorithms | `aed9ad5` | ✅ |
| 18 | 5-fold cross-validation — two best models | `95f91f9` | ✅ |
| 19 | Best model diagnostics | `9fd30a6` | ✅ |
| 20 | Final findings & conclusion | `22a8a90` | ✅ |
| 21 | Notebook consolidation — single `regression.ipynb`, portable dataset path | `fa04bb5` | ✅ |

### 🚗 Classification Track — `notebooks/classification.ipynb`

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 00 | Notebook created & initial commit | `02cdc2d`, `c6a3ef3` | ✅ |
| 01 | Data loading | `3a3e00a` | ✅ |
| 02 | Dataset audit | `6234a46` | ✅ |
| 03 | EDA analysis | `0c3399f` | ✅ |
| 04 | Data cleaning | `3177642` | ✅ |
| 05 | Feature scaling & engineering — `Vehicle Age` | `038fd9c` | ✅ |
| 06 | Stratified train/test split + 60k subset | `02044fc` | ✅ |
| 07 | Encoding — OneHot + StandardScaler | `57c42dc` | ✅ |
| 08 | Algorithm 1 — Logistic Regression | `72b9809` | ✅ |
| 09 | Algorithm 2 — K-Nearest Neighbors | `6885be9` | ✅ |
| 10 | Algorithm 3 — Gaussian Naive Bayes | `9b46c72` | ✅ |
| 11 | Algorithm 4 — Decision Tree Classifier | `61f3d80` | ✅ |
| 12 | Algorithm 5 — Support Vector Machine | `33199d7` | ✅ |
| 13 | Comparison of all 5 Part-A algorithms | `4fd67a7` | ✅ |
| 14 | Conclusion | `0356b4f` | ✅ |

### 📄 Documentation

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 01 | README rewritten — both tracks, separate pipelines | `260e7c6` | ✅ |
| 02 | PROGRESS.md — this board | *pending* | 🚧 |

> ✅ **49 granular commits, no bulk upload.** The guidelines state that a single commit containing all work scores **zero** on the GitHub criterion — the current history comfortably satisfies it.

---

## 🔋 Regression Track — 10 / 10 Algorithms

**Target:** `SOH_pct` (%) · **Dataset:** NASA Li-ion Battery Aging (PCoE) · **Split:** `GroupShuffleSplit` on `battery_id` — train 24 batteries / 2,345 rows, test **6 fully unseen batteries** / 349 rows, `random_state=42`. RMSE and MAE are in SOH percentage points.

| # | Algorithm | Best parameters | R² | RMSE | MAE | Status |
|:-:|:--|:--|--:|--:|--:|:-:|
| 1 | Linear Regression | *baseline* | 0.6671 | 10.8000 | 8.3770 | ✅ |
| 2 | Ridge Regression | `alpha=100` | 0.6650 | 10.8331 | 8.8837 | ✅ |
| 3 | Lasso Regression | `alpha=0.5` *(zeroed 4/10 coefficients)* | 0.6820 | 10.5553 | 8.6171 | ✅ |
| 4 | ElasticNet Regression | `alpha=0.1, l1_ratio=0.1` | 0.6558 | 10.9816 | 9.1909 | ✅ |
| 5 | Polynomial Regression | `degree=2` *(GroupKFold-selected)* | 0.7862 | 8.6539 | 5.9320 | ✅ |
| 6 | Decision Tree Regressor | `max_depth=3, min_samples_leaf=1, min_samples_split=2` | 0.6525 | 11.0345 | 8.3325 | ✅ |
| 7 | Random Forest Regressor | `max_depth=None, min_samples_leaf=4, n_estimators=200` | 0.7924 | 8.5277 | 5.5031 | ✅ |
| 8 | **Gradient Boosting Regressor** | `learning_rate=0.1, max_depth=3, n_estimators=200` | **0.8153** | **8.0449** | 5.6427 | ✅ |
| 9 | Support Vector Regressor | `C=100, epsilon=0.1, kernel=rbf` | 0.7919 | 8.5396 | 6.5133 | ✅ |
| 10 | K-Nearest Neighbors Regressor | `n_neighbors=7, p=1, weights=distance` | 0.8013 | 8.3445 | **5.1617** | ✅ |

**Best model:** Gradient Boosting — R² 0.8153, MAE 5.64 SOH-points on unseen batteries.

### Mandatory metric status

| Metric | Required for | Status |
|:--|:--|:-:|
| R² | all 10 models | ✅ 10/10 |
| RMSE | all 10 models | ✅ 10/10 |
| MAE | all 10 models | ✅ 10/10 |
| **5-fold cross-validated R²** | the two best-performing models | ✅ done |

**5-fold `GroupKFold` CV** (grouped on `battery_id`, so folds never split a battery):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | 0.8998 | 0.9209 | 0.6566 | 0.8654 | 0.7764 | **0.8238** | 0.0971 |
| KNN | 0.8242 | 0.6071 | 0.7442 | 0.6381 | 0.6897 | **0.7006** | 0.0774 |

### Hyperparameter tuning

`GridSearchCV` (5-fold) applied to **8 models** — Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN. Polynomial degree is selected separately via `GroupKFold` + `cross_val_score` on the training data only. Before/after on the held-out test set:

| Model | Original R² | Tuned R² | Δ | Note |
|:--|--:|--:|--:|:--|
| Gradient Boosting | 0.8053 | 0.8153 | **+0.0100** | Improvement documented — satisfies C3 |
| Decision Tree | 0.6672 | 0.6525 | −0.0148 | Non-improvement, documented rather than hidden |

### Required visualisations

| Plot | Where | Status |
|:--|:--|:-:|
| Predicted vs. actual (best model) | §14 | ✅ |
| Residual plot (best model) | §14 | ✅ |
| Feature importance (tree-based model) | §11 — Decision Tree | ✅ |

---

## 🚗 Classification Track — Part A — 5 / 5 Algorithms

> **Separate dataset.** This track shares nothing with the Regression track — different data, different preprocessing, different target.

**Target:** `Electric Vehicle Type` — binary, BEV **80.75 %** / PHEV **19.25 %**
**Dataset:** Washington State EV Population Data, **294,193 × 16** — `data/Classification_Dataset/Electric_Vehicle_Population_Data.csv`
**Split:** stratified 80:20 (train 235,354 / test 58,839), then a stratified **60,000-row training subset**; `OneHotEncoder` + `StandardScaler` fitted on training data only.

| # | Algorithm | Tuning | Accuracy | Precision | Recall | Weighted F1 | Status |
|:-:|:--|:--|--:|--:|--:|--:|:-:|
| 1 | Logistic Regression | `max_iter=1000` *(baseline)* | 0.9933 | 0.9933 | 0.9933 | 0.9933 | ✅ |
| 2 | K-Nearest Neighbors | `k=3`, Euclidean *(validation-tuned)* | 0.9725 | 0.9722 | 0.9725 | 0.9723 | ✅ |
| 3 | Gaussian Naive Bayes | — | 0.9867 | 0.9874 | 0.9867 | 0.9868 | ✅ |
| 4 | **Decision Tree Classifier** | `max_depth=7` *(validation-tuned)* | **0.9999** | **0.9999** | **0.9999** | **0.9999** | ✅ |
| 5 | Support Vector Machine (SVC) | `kernel=rbf, C=1.0` *(grid-tuned)* | 0.9975 | 0.9975 | 0.9975 | 0.9975 | ✅ |

**Best model:** Decision Tree, on both accuracy and weighted F1.

| Review-1 requirement | Status |
|:--|:-:|
| All 5 Part-A algorithms trained, no errors | ✅ |
| Accuracy reported per algorithm | ✅ |
| Weighted F1 reported per algorithm | ✅ |
| Confusion matrix per algorithm | ✅ 5/5 |
| Preliminary comparison table | ✅ §14 + bar chart |
| Decision tree visualised | ✅ first 3 levels |

> ⚠️ **ROC-AUC is not yet computed.** It is a **Review-2** requirement (§3.2), not Review 1, so this is on schedule rather than overdue.

---

## 🧮 Rubric Scoreboard

### Review 1 — 25 marks

| § | Criterion | Max | Status | Note |
|:-:|:--|--:|:-:|:--|
| A1 | Dataset loading & audit | 1 | ✅ | Shape, dtypes, missing counts, target distribution — both notebooks |
| A2 | EDA visualisations | 2 | ✅ | Distributions, correlation heatmap, target distribution, 2+ feature–target scatters — both notebooks |
| A3 | Insight commentary | 1 | ✅ | Markdown observation after every major figure |
| B1 | Data cleaning | 1 | ✅ | Justified strategy in both tracks; duplicates and outliers addressed |
| B2 | Encoding, scaling & splitting | 1 | ✅ | Train-fit-only scalers; grouped split (regression), stratified split (classification) |
| B3 | Feature engineering | 1 | ✅ | `cycle_index_squared` · `Vehicle Age` — both justified in writing |
| C1 | All 10 regression algorithms | 4 | ✅ | 10/10 trained, no errors |
| C2 | Comparative evaluation table | 2 | ✅ | Single table, all 10 models, ranked by R² |
| C3 | Hyperparameter tuning (≥ 2 models) | 2 | ✅ | GridSearchCV on 8 models; GB and DT deltas reported |
| C4 | Visualisations (residual · pred-vs-actual · importance) | 1 | ✅ | All three present |
| D1 | Classification Part A — 5 algorithms | 2 | ✅ | 5/5 trained, no errors |
| D2 | Classification Part A — evaluation | 1 | ✅ | Accuracy, weighted F1, confusion matrix + comparison table |
| E1 | Presentation quality | 1 | ⬜ | Rehearse the narrative end to end |
| — | **Viva** | 5 | ⬜ | All three members must be able to explain any cell |
| | **Implementation subtotal** | **19** | **✅ 19/19** | **every code deliverable is in the repository** |
| | **Total** | **25** | **🚧** | **6 marks decided on presentation day** |

### Review 2 — 25 marks (+2 bonus)

| § | Criterion | Max | Status |
|:-:|:--|--:|:-:|
| A1 | Classification Part B — 5 algorithms | 3 | ⬜ |
| A2 | Full 10-algorithm comparison table | 1 | ⬜ |
| A3 | Final model selection + tuning | 2 | ⬜ |
| B1 | Clustering implementation (both algorithms) | 4 | ⬜ |
| B2 | Silhouette / DB / CH + Elbow + Dendrogram | 2 | ⬜ |
| B3 | PCA-2D per algorithm + t-SNE for one | 2 | ⬜ |
| C1 | Code quality | 1 | 🚧 |
| C2 | README & documentation | 1 | ✅ |
| C3 | GitHub history + `requirements.txt` | 1 | ✅ |
| D | Presentation & Viva | 8 | ⬜ |
| — | Bonus — GUI / deployment | +2 | ⬜ |

---

## 📦 Backlog — Review 2

### 🚗 Classification Part B *(6 marks)*

- [ ] Random Forest Classifier — feature importance, tune `n_estimators`
- [ ] AdaBoost Classifier — tune `n_estimators` and `learning_rate`
- [ ] Gradient Boosting Classifier — sklearn GBM or XGBoost / LightGBM
- [ ] Bagging Classifier — Decision Tree as base estimator
- [ ] MLP Classifier — tune `hidden_layer_sizes` and `activation`
- [ ] **Single consolidated table across all 10 classification algorithms** — Accuracy, Precision, Recall, F1, ROC-AUC
- [ ] **ROC-AUC** — the target is binary, so a standard ROC-AUC applies; OvR is only needed if the target is ever changed to a multi-class one
- [ ] Final model selection with justification + tuning + documented improvement

### 🔌 Clustering Track *(8 marks)*

- [ ] Obtain the EV Charging Session dataset and place it under `data/`
- [ ] Create `notebooks/clustering.ipynb`
- [ ] K-Means — **Elbow curve** (inertia vs. k) to choose k
- [ ] Agglomerative Hierarchical Clustering — **Dendrogram**, compare linkage strategies
- [ ] Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index for **both** algorithms
- [ ] **PCA-reduced 2D scatter with cluster colour-coding for every algorithm** *(mandatory)*
- [ ] t-SNE plot for at least one algorithm
- [ ] ⚠️ **No labels during fitting.** Any ground-truth/site-type field may be used *only afterwards* to interpret clusters.

### 🧩 Pipeline Integration & Quality *(3 marks)*

- [x] README with dataset description, problem statement, results table, environment setup and how-to-run *(C2)* — [README.md](README.md)
- [x] `requirements.txt` present with pinned versions *(C3)* — [requirements.txt](requirements.txt)
- [ ] All notebooks run top-to-bottom without errors *(C1)* — ✅ regression, ✅ classification Part A; pending the Part B and clustering work

### 🎁 Bonus *(up to +2)*

- [ ] GUI / deployed application in `app/` — e.g. a Streamlit page that takes cycle features and returns a predicted SOH %

---

## ⚠️ Risks & Open Items

| | Item | Impact | Action |
|:-:|:--|:--|:--|
| 🟡 | **`a2_*.png` EDA figures modified but uncommitted** — their bytes reverted to the earlier `befa20d` renderings | README embeds all three; the repo and the figures could drift apart | Decide whether to keep or regenerate, then commit |
| 🟡 | **Classification scores are near-perfect** (Decision Tree 0.9999) | Not a bug — `Model`, `Make` and `Electric Range` are near-deterministic proxies for BEV vs PHEV even after the CAFV column is dropped | Expect a viva question; the Review-2 ROC-AUC work is the natural place to probe it |
| 🟡 | **Severe feature collinearity** in the regression track — `ambient_temperature` correlates 0.94 with `charge_max_temp`, 0.82 with `charge_time_to_cv`, −0.78 with `Rct` | Linear coefficients cannot be read as isolated effects; some signs flip | Already documented in §C1.1; be ready to explain |
| 🟡 | **Clustering dataset not yet in `data/`** | Blocks the entire Review-2 clustering track (8 marks) | Source the EV Charging Session dataset early |

---

## 🔁 Keeping This File Current

Update this board with **every `feat:` commit** — tick the milestone, fill in the algorithm's metrics from the notebook output, and adjust the badges at the top. Where a notebook is re-run, re-read its outputs rather than trusting earlier figures: several metrics shifted between runs during Review-1 preparation.

<div align="center">
<br>
<sub>⚡ <b>ChargeIQ</b> · 23CSE301 Machine Learning Capstone · AY 2026–27 · <a href="README.md">back to README</a></sub>
</div>
