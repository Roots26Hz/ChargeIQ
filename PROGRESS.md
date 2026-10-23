<div align="center">

# 📍 ChargeIQ — Progress Board

**Live status of the 23CSE301 Machine Learning capstone, tracked against the official rubric.**

<br>

![Overall](https://img.shields.io/badge/Deliverables%20in%20repo-36%2F50%20marks-FFB300?style=for-the-badge)
![Review 2](https://img.shields.io/badge/Review%202-17%2F25-FFB300?style=for-the-badge)
[![Bonus GUI](https://img.shields.io/badge/🎁%20Bonus%20GUI-Live%20·%20%2B2-2E7D32?style=flat-square)](https://chargeiq.streamlit.app)

<img src="assets/badges/badge-review1-19-19.svg" alt="Review 1 implementation: 19 of 19" height="28">
<img src="assets/badges/badge-review2-17-17.svg" alt="Review 2 implementation: 17 of 17" height="28">
<img src="assets/badges/badge-regression-10-10.svg" alt="Regression: 10 of 10 algorithms" height="28">
<img src="assets/badges/badge-classification-a-5-5.svg" alt="Classification Part A: 5 of 5" height="28">
<img src="assets/badges/badge-classification-b-5-5.svg" alt="Classification Part B: 5 of 5" height="28">
<img src="assets/badges/badge-clustering-2-2.svg" alt="Clustering: 2 of 2 algorithms" height="28">
<img src="assets/badges/badge-bonus-live.svg" alt="Bonus GUI and deployment: Live" height="28">

<br>

<img src="assets/badges/bar-regression.svg" alt="Regression algorithms: 10 of 10 complete" width="640"><br>
<img src="assets/badges/bar-classification-a.svg" alt="Classification Part A algorithms: 5 of 5 complete" width="640"><br>
<img src="assets/badges/bar-classification-b.svg" alt="Classification Part B algorithms: 5 of 5 complete" width="640"><br>
<img src="assets/badges/bar-clustering.svg" alt="Clustering algorithms: 2 of 2 complete" width="640"><br>
<img src="assets/badges/bar-marks.svg" alt="Deliverables in repository: 36 of 50 marks" width="640">

**🖥️ Live app: [chargeiq.streamlit.app](https://chargeiq.streamlit.app)**

*Last updated: 23 October 2026 · 94 commits · at [`99ef597`](https://github.com/Roots26Hz/ChargeIQ/commit/99ef597)*

</div>

> **Legend** — ✅ done · 🚧 in progress · ⬜ not started · ⚠️ at risk
>
> **Badge meaning** — marks whose deliverable is physically in the repository. Review 1: **19/19** implementation marks done; the remaining 6 (presentation + viva) are earned on the day. Review 2: **17/25** — Part B (A1–A3), clustering (B1–B3), code quality (C1), README (C2) and history + `requirements.txt` (C3); presentation + viva (D, 8 marks) are earned on the day. The **+2 bonus** (GUI and public deployment) is delivered.

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
| 15 | Part B · Algorithm 6 — Random Forest | `a9624c2` | ✅ |
| 16 | Part B · Algorithm 7 — AdaBoost | `ccc98b1` | ✅ |
| 17 | Part B · Algorithm 8 — Gradient Boosting | `f6385b2` | ✅ |
| 18 | Part B · Algorithm 9 — Bagging | `ccfa62e` | ✅ |
| 19 | Part B · Algorithm 10 — MLP | `4f369e6` | ✅ |
| 20 | Part B · confusion matrices and ROC curves | `e669912` | ✅ |
| 21 | Part B · evaluation — 10-model comparison, 5-fold CV | `2a74064` | ✅ |
| 22 | Final selection and conclusion | `bc5abdf` | ✅ |

### 🔌 Clustering Track — `notebooks/clustering.ipynb`

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 00 | Notebook created | `02accf7` … `68f6075` | ✅ |
| 01 | Preprocessing — sample, imputation, Yeo–Johnson, scaling, elbow curve | `ccc4c83` | ✅ |
| 02 | Linkage comparison and dendrograms | `e56591a` | ✅ |
| 03 | Evaluation — Silhouette, Davies–Bouldin, Calinski–Harabasz | `14ec696` | ✅ |
| 04 | Visualisation — PCA for both algorithms, t-SNE, cluster profiles | `2e9d40f` … `171ad12` | ✅ |

### 🖥️ Live App — `app/` and `models/` *(bonus)*

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 01 | Vehicle model rebuilt from the notebook and verified; inference module + tests | `2bee195` | ✅ |
| 02 | Streamlit skeleton — navigation, design system, theme | `a1ef325` | ✅ |
| 03 | Vehicle story chapters | `f87bdcb` | ✅ |
| 04 | Vehicle prediction chapter | `633ec7c` | ✅ |
| 05 | Team notes · deployment readiness · **first public deploy** | `49e92c9`, `906b142` | ✅ |
| 06 | Battery model rebuilt and verified | `584ff16` | ✅ |
| 07 | Battery chapters | `a4e2cad` | ✅ |
| 08 | Clustering model rebuilt and verified; 8 clustering figures exported | `35ef591` | ✅ |
| 09 | Driver chapters | `96f281a`, `cde89d9` | ✅ |
| 10 | Conclusion chapter, polish, redeploy — all 11 chapters live | `35608e6` | ✅ |

### 📄 Documentation & Repository Hygiene

| # | Milestone | Commit | Status |
|:-:|:--|:--|:-:|
| 01 | README rewritten — both tracks, separate pipelines, results tables | `260e7c6` | ✅ |
| 02 | PROGRESS.md — this board | `5aeec13` | ✅ |
| 03 | README correction — GridSearchCV model count | `571fbbd` | ✅ |
| 04 | Repository structure alignment — duplicate notebook removed from `HEAD`, `.DS_Store` untracked, `requirements.txt` pinned | `73c5cd1` | ✅ |
| 05 | Figure gallery — 27 figures saved to `assets/figures/`, README gallery expanded, classification dataset path fixed | `e33bd0d` | ✅ |
| 06 | Team table — GitHub IDs column, all three members filled in | `bba01ae` | ✅ |
| 07 | Folder READMEs — `notebooks/` and `data/` documented; stale dataset copy removed | `bba01ae` | ✅ |
| 08 | README and PROGRESS.md updated for Review 2 — Part B, clustering, the live app, AI disclosure; folder READMEs refreshed | `99ef597` | ✅ |
| 09 | `clustering.ipynb` reads its dataset from `../data/Clustering_Dataset/` — C1 complete | *this update* | ✅ |

> ✅ **94 granular commits, no bulk upload.** The guidelines state that a single commit containing all work scores **zero** on the GitHub criterion — the current history comfortably satisfies it.

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

| Plot | Where | Saved to | Status |
|:--|:--|:--|:-:|
| Predicted vs. actual (best model) | §14 | `assets/figures/regression/models/02_predicted_vs_actual.png` | ✅ |
| Residual plot (best model) | §14 | `assets/figures/regression/models/03_residual_plot.png` | ✅ |
| Feature importance (tree-based model) | §11 — Decision Tree | `assets/figures/regression/models/01_decision_tree_feature_importance.png` | ✅ |

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
| 3 | Gaussian Naive Bayes | — *(15,000-row stratified subset)* | 0.9834 | 0.9841 | 0.9834 | 0.9835 | ✅ |
| 4 | **Decision Tree Classifier** | `max_depth=7` *(validation-tuned)* | **0.9999** | **0.9999** | **0.9999** | **0.9999** | ✅ |
| 5 | Support Vector Machine (SVC) | `kernel=rbf, C=1.0` *(grid-tuned)* | 0.9975 | 0.9975 | 0.9975 | 0.9975 | ✅ |

**Best model:** Decision Tree, on both accuracy and weighted F1.

| Review-1 requirement | Status |
|:--|:-:|
| All 5 Part-A algorithms trained, no errors | ✅ |
| Accuracy reported per algorithm | ✅ |
| Weighted F1 reported per algorithm | ✅ |
| Confusion matrix per algorithm | ✅ 5/5 — `assets/figures/classification/models/` |
| Preliminary comparison table | ✅ §14 + bar chart |
| Decision tree visualised | ✅ first 3 levels |

> ✅ **ROC-AUC** — computed for all ten classifiers in Part B (cell 81); see below.

---

## 🧩 Classification Track — Part B — 5 / 5 Algorithms

Same 60,000-row training subset, same 58,839-row test set, same weighted metrics, plus **ROC-AUC**.

| # | Algorithm | Configuration | Accuracy | Weighted F1 | ROC-AUC | Status |
|:-:|:--|:--|--:|--:|--:|:-:|
| 6 | Random Forest | `n_estimators=100, max_depth=None` *(grid-tuned on a 20,000-row subset)* | 0.9988 | 0.9988 | 1.0000 | ✅ |
| 7 | AdaBoost | stumps; `n_estimators=50, learning_rate=0.5` *(grid-tuned)* | 0.9994 | 0.9994 | 1.0000 | ✅ |
| 8 | Gradient Boosting | `n_estimators=100, learning_rate=0.1, max_depth=3` on 100 `TruncatedSVD` components | 0.9928 | 0.9928 | 0.9995 | ✅ |
| 9 | Bagging | 50 trees (`max_depth=15`), `max_samples=0.8` | **0.9999** | **0.9999** | 0.9999 | ✅ |
| 10 | MLP | `(100, 50)`, ReLU, Adam, early stopping — 20 iterations | 0.9978 | 0.9978 | 1.0000 | ✅ |

| Review-2 requirement | Status |
|:--|:-:|
| All 5 Part-B algorithms trained, no errors | ✅ |
| Single table across all 10 classifiers — Accuracy, Precision, Recall, F1, ROC-AUC | ✅ cell 81 |
| Confusion matrix and ROC curve per Part-B model | ✅ cells 77–78 |
| 5-fold CV of the two best — Decision Tree 0.9998 ± 0.0004, Bagging 0.9995 ± 0.0004 | ✅ cell 84 |
| Final model selected with justification — **Decision Tree** (best weighted F1 and accuracy; AdaBoost best ROC-AUC after rounding) | ✅ cells 86–87 |

> 🔎 **Leakage audit, completed in the app.** All 192,477 registrations with `Electric Range = 0` are BEVs labelled *"battery range has not been researched"* — the same information as the dropped CAFV column, and the tree's first rule. See [Chapter 3](https://chargeiq.streamlit.app/vehicle-model).

---

## 🔌 Clustering Track — 2 / 2 Algorithms

**Dataset:** synthetic EV charging sessions — Gholizadeh (2024), *Electric Vehicle Charging Dataset*, Mendeley Data, [doi:10.17632/5zrtmp7gwd.1](https://doi.org/10.17632/5zrtmp7gwd.1), CC BY 4.0 · **1,965,239 × 4**, of which a reproducible **100,000-session sample** is clustered · **Features:** connection time, charging duration, energy delivered · **Preprocessing:** median imputation → Yeo–Johnson → `StandardScaler` · **K = 3** from the elbow curve.

| # | Algorithm | Configuration | Silhouette ⬆ | Davies–Bouldin ⬇ | Calinski–Harabasz ⬆ | Status |
|:-:|:--|:--|--:|--:|--:|:-:|
| 1 | **K-Means** | `n_clusters=3, n_init=5, max_iter=100` | **0.4094** | **0.9195** | **84,854.2** | ✅ |
| 2 | Agglomerative | Ward linkage, 10-nearest-neighbour connectivity | 0.3898 | 0.9767 | 73,872.0 | ✅ |

K-Means ranks first on all three metrics. Group sizes: K-Means 35,099 · 46,195 · 18,706; Agglomerative 49,776 · 18,803 · 31,421.

| Review-2 requirement | Status |
|:--|:-:|
| Both algorithms fitted, labels obtained | ✅ cells 15, 21 |
| Silhouette, Davies–Bouldin, Calinski–Harabasz for both | ✅ cells 23, 33 |
| Elbow curve | ✅ cell 13 |
| Dendrograms — single, complete, average, Ward | ✅ cells 17, 19 |
| PCA-2D scatter for **every** algorithm *(mandatory)* | ✅ cells 25, 27 — 88.53 % of variance |
| t-SNE for at least one algorithm | ✅ cell 29 — K-Means, 1,000 sessions |
| No labels during fitting | ✅ |

---

## 🖥️ Live App — Bonus

**[chargeiq.streamlit.app](https://chargeiq.streamlit.app)** — eleven chapters, three tracks, each told as *meet the data → how the model works → try it yourself*, ending in a conclusion.

| Bonus requirement | Status |
|:--|:-:|
| GUI that accepts user inputs and returns predictions from the trained models | ✅ all three models — vehicle type, State of Health, charging group |
| Accessible via a public URL | ✅ Streamlit Community Cloud, verified from a signed-out session |
| The notebooks' own models | ✅ rebuilt by `models/export_*.py` and written only after matching the notebooks' stored results |
| Automated tests | ✅ 259 — models, every quote and figure against the notebooks, deployment readiness |

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

| § | Criterion | Max | Status | Note |
|:-:|:--|--:|:-:|:--|
| A1 | Classification Part B — 5 algorithms | 3 | ✅ | RF, AdaBoost, Gradient Boosting, Bagging, MLP |
| A2 | Full 10-algorithm comparison table | 1 | ✅ | Accuracy, Precision, Recall, weighted F1, ROC-AUC — cell 81 |
| A3 | Final model selection + tuning | 2 | ✅ | Decision Tree; RF and AdaBoost grids; 5-fold CV |
| B1 | Clustering implementation (both algorithms) | 4 | ✅ | K-Means + Agglomerative, K = 3 |
| B2 | Silhouette / DB / CH + Elbow + Dendrogram | 2 | ✅ | Both algorithms; four linkages |
| B3 | PCA-2D per algorithm + t-SNE for one | 2 | ✅ | PCA for both, t-SNE for K-Means |
| C1 | Code quality | 1 | ✅ | All three notebooks use repo-relative dataset paths and run top to bottom |
| C2 | README & documentation | 1 | ✅ | README, PROGRESS.md, folder READMEs, `app/DEPLOYMENT.md` |
| C3 | GitHub history + `requirements.txt` | 1 | ✅ | 94 commits; pinned requirements |
| D | Presentation & Viva | 8 | ⬜ | On the day — see the review walkthrough |
| | **Implementation subtotal** | **17** | **✅ 17/17** | **every code deliverable is in the repository** |
| ⭐ | Bonus — GUI / deployment | +2 | ✅ | [chargeiq.streamlit.app](https://chargeiq.streamlit.app) |

---

## 📦 Backlog — Review 2

### 🚗 Classification Part B *(6 marks)*

- [x] Random Forest Classifier — feature importance, tuned `n_estimators` × `max_depth`
- [x] AdaBoost Classifier — tuned `n_estimators` and `learning_rate`
- [x] Gradient Boosting Classifier — scikit-learn, on 100 `TruncatedSVD` components
- [x] Bagging Classifier — Decision Tree as base estimator
- [x] MLP Classifier — `(100, 50)`, ReLU, early stopping
- [x] **Single consolidated table across all 10 classification algorithms** — Accuracy, Precision, Recall, F1, ROC-AUC
- [x] **ROC-AUC** for every model, with ROC curves for Part B
- [x] Final model selection with justification + tuning + 5-fold CV

### 🔌 Clustering Track *(8 marks)*

- [x] Obtain the EV charging-session dataset and place it under `data/Clustering_Dataset/`
- [x] Create `notebooks/clustering.ipynb`
- [x] K-Means — **Elbow curve** (inertia vs. k) to choose k
- [x] Agglomerative Hierarchical Clustering — **Dendrogram**, compare linkage strategies
- [x] Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index for **both** algorithms
- [x] **PCA-reduced 2D scatter with cluster colour-coding for every algorithm** *(mandatory)*
- [x] t-SNE plot for at least one algorithm
- [x] No labels during fitting

### 🧩 Pipeline Integration & Quality *(3 marks)*

- [x] README with dataset description, problem statement, results table, environment setup and how-to-run *(C2)* — [README.md](README.md)
- [x] `requirements.txt` present with pinned versions *(C3)* — [requirements.txt](requirements.txt), plus [app/requirements.txt](app/requirements.txt)
- [x] Single notebook per track; Guidelines-mandated repo layout; `.DS_Store` untracked *(C1/C3)*
- [x] Folder-level documentation — [`notebooks/`](notebooks/README.md), [`data/`](data/README.md), [`app/`](app/README.md), [`models/`](models/README.md)
- [x] All notebooks run top-to-bottom without errors *(C1)* — regression, classification and clustering, each reading its dataset from `../data/`

### 🎁 Bonus *(up to +2)*

- [x] GUI in `app/` — all three trained models, each with its own interactive chapter
- [x] Deployed at a public URL — [chargeiq.streamlit.app](https://chargeiq.streamlit.app)

---

## ⚠️ Risks & Open Items

| | Item | Impact | Action |
|:-:|:--|:--|:--|
| 🟡 | **Classification scores are near-perfect** (Decision Tree 0.9999) | Explained: `Electric Range = 0` means "range not researched" and carries the CAFV information that was dropped | Shown openly in [app Chapter 3](https://chargeiq.streamlit.app/vehicle-model); expect a viva question |
| 🟡 | **Regression tuning used ordinary 5-fold CV** (cell 61), so cycles of one battery can sit in training and validation folds | Tuning scores are optimistic; the held-out evaluation on 6 unseen batteries and the GroupKFold CV (cell 78) are unaffected | Acknowledge in the viva; grouping the tuning folds is the natural next step |
| 🟡 | **Severe feature collinearity** in the regression track — `ambient_temperature` correlates 0.94 with `charge_max_temp`, 0.82 with `charge_time_to_cv`, −0.78 with `Rct` | Linear coefficients cannot be read as isolated effects; some signs flip | Already documented in §C1.1; be ready to explain |
| 🟡 | **Some figures are not in `assets/figures/`** — Part B's confusion matrices, ROC curves, RF importances and comparison chart; the saved Naive Bayes confusion matrix is from an earlier run | The README gallery cannot show them | Add `plt.savefig` calls to those cells and re-run |
| 🟡 | **The app sleeps after 12 hours without visitors** | A wake-up screen on first visit | Open [the app](https://chargeiq.streamlit.app) shortly before each review |
| 🟢 | **Dataset subfolders cannot carry their own README** — `.gitignore` excludes `/data/*_Dataset` wholesale | A per-folder README would never reach GitHub | All three are documented in the committable [`data/README.md`](data/README.md) |
---

## 🔁 Keeping This File Current

Update this board with **every `feat:` commit** — tick the milestone, fill in the algorithm's metrics from the notebook output, and adjust the badges at the top. Where a notebook is re-run, re-read its outputs rather than trusting earlier figures: several metrics shifted between runs during Review-1 preparation.

<div align="center">
<br>
<sub>⚡ <b>ChargeIQ</b> · 23CSE301 Machine Learning Capstone · AY 2026–27 · <a href="README.md">back to README</a></sub>
</div>
