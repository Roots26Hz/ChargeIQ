# 📓 `notebooks/`

The three Jupyter notebooks required by the 23CSE301 capstone guidelines — one per problem track. Each is self-contained: its own dataset, its own preprocessing, its own target and its own evaluation. Nothing is shared between them.

Results tables and the figure gallery live in the [root README](../README.md); mark-by-mark status is in [`PROGRESS.md`](../PROGRESS.md).

## Contents

| File | Track | Target | Algorithms | Status |
|:--|:--|:--|:-:|:-:|
| `regression.ipynb` | 🔋 Regression | `SOH_pct` — State of Health (%) | 10 / 10 | ✅ |
| `classification.ipynb` | 🚗 Classification — Part A + Part B | `Electric Vehicle Type` — BEV vs PHEV | 10 / 10 | ✅ |
| `clustering.ipynb` | 🔌 Clustering | *none — unsupervised* | 2 / 2 | ✅ |

Two data files sit beside `clustering.ipynb`, written by its last cell: `EV_clustering_sample_with_labels.csv` (the exact 100,000-session sample it clustered, with both algorithms' labels) and `EV_clustering_evaluation_metrics.csv` (both algorithms' metrics).

---

## 🔋 `regression.ipynb`

Predicts battery **State of Health** from the NASA Li-ion Battery Aging dataset. 84 cells, sections 0 → 15.

| § | Contents |
|:--|:--|
| 0 | Imports, `random_state=42`, dataset and figure paths |
| 1 | Load & audit `metadata.csv`; structural join to one row per discharge cycle |
| 2 | Feature extraction from the raw per-cycle curve CSVs |
| 3 | Exploratory data analysis *(3 saved figures)* |
| 4 | Data cleaning — corrupted cells dropped, gaps imputed, outliers capped |
| 5 | Target engineering — `SOH_pct = 100 × Capacity / NOMINAL_CAPACITY` |
| 6–7 | Feature set, battery-wise `GroupShuffleSplit`, `StandardScaler` |
| 8 | Feature engineering — `cycle_index_squared` |
| 9 | Algorithms 1–5 — Linear, Ridge, Lasso, ElasticNet, Polynomial |
| 10 | First comparison table |
| 11 | Algorithms 6–10 — Decision Tree, Random Forest, Gradient Boosting, SVR, KNN *(1 saved figure)* |
| 12 | Final comparison — all 10 algorithms, ranked by R² |
| 13 | 5-fold `GroupKFold` cross-validation on the two best models |
| 14 | Best-model diagnostics *(2 saved figures)* |
| 15 | Findings and conclusion |

**Key design decision:** the split is grouped on `battery_id`, so every cycle from a given cell goes entirely into train *or* test. `battery_id` is not used as a feature. The reasoning is written up in §6 and in the root README.

**Dataset path.** Defaults to `../data/Regression_Dataset`. To use a copy elsewhere, set an environment variable instead of editing the notebook:

```bash
export CHARGEIQ_REGRESSION_DATA="/path/to/your/cleaned_dataset"
```

---

## 🚗 `classification.ipynb`

Predicts **`Electric Vehicle Type`** (BEV vs PHEV) from the Washington State EV Population dataset. 88 cells: Part A (five classifiers, Review 1) and Part B (five more, Review 2).

| § | Contents |
|:--|:--|
| 2–3 | Imports, figure paths, data loading |
| 3 | Dataset audit — shape, dtypes, missing values, duplicates, class balance |
| 4 | EDA — distributions, target distribution, correlation heatmap, feature–target scatters *(13 saved figures)* |
| 5 | Data cleaning — median/mode imputation; IQR outliers identified but retained |
| 6 | Feature selection and engineering — `Vehicle Age = 2026 − Model Year` |
| 7 | Stratified 80:20 split, then a stratified 60,000-row training subset |
| 8 | `OneHotEncoder` + `StandardScaler` via `ColumnTransformer`, fitted on training data only |
| 9–13 | Algorithms 1–5 — Logistic Regression, KNN, Gaussian NB, Decision Tree, SVM *(8 saved figures)* |
| 14 | Comparison of all five, with bar chart |
| Part B, 6–10 | Random Forest (tuned, with feature importances), AdaBoost (tuned), Gradient Boosting (on 100 `TruncatedSVD` components), Bagging, MLP |
| Part B | Confusion matrices and ROC curves; the 10-classifier comparison with ROC-AUC (cell 81); 5-fold CV of the two best (cell 84) |
| Part B | Final model selection and conclusion — Decision Tree (cells 86–87) |

**Three columns are dropped before modelling**, for two different reasons: `VIN (1-10)` and `DOL Vehicle ID` are identifiers, and `Clean Alternative Fuel Vehicle (CAFV) Eligibility` is excluded as a **target proxy** — it is derived from vehicle type and electric range, so leaving it in would leak the answer.

Reads `../data/Classification_Dataset/Electric_Vehicle_Population_Data.csv`.

---

## 🔌 `clustering.ipynb`

Groups EV charging sessions — when they start, how long they last, how much energy they deliver — on a **synthetic** dataset (see [`../data/README.md`](../data/README.md)). 38 cells. No labels are used during fitting.

| Cells | Contents |
|:--|:--|
| 3–9 | Imports; load; audit; cleaning — 1,965,239 sessions, all retained |
| 10–11 | EDA — distributions and correlation heatmap, on 10,000 sessions |
| 12–13 | A reproducible 100,000-session sample; median imputation → Yeo–Johnson → `StandardScaler`; elbow curve for K = 2…10 |
| 14–15 | K-Means, K = 3 |
| 16–21 | Single, complete, average and Ward dendrograms (500 sessions); Agglomerative clustering, Ward + 10-nearest-neighbour connectivity |
| 22–23, 32–33 | Silhouette (5,000-session subset), Davies–Bouldin and Calinski–Harabasz for both, ranked |
| 24–29 | PCA-2D for both algorithms; t-SNE for K-Means (1,000 sessions) |
| 30–31 | Cluster profiles in original units |
| 34–36 | Export of the sample and metrics; conclusion |

---

## 🖼️ Figures

The regression notebook and Part A of the classification notebook write their figures to [`../assets/figures/`](../assets/figures) with a `plt.savefig` call in the cell that draws them, so **Run All** regenerates them. `clustering.ipynb` only displays its plots; its 8 figures were extracted byte for byte from the stored outputs by [`../models/export_clustering.py`](../models/export_clustering.py). Classification Part B's figures appear in the notebook only.

```
assets/figures/regression/eda/          3 figures
assets/figures/regression/models/       3 figures
assets/figures/classification/eda/     13 figures
assets/figures/classification/models/   8 figures
assets/figures/clustering/eda/          2 figures
assets/figures/clustering/models/       6 figures
```

---

## ▶️ Running

From the repository root:

```bash
pip install -r ../requirements.txt
```

```bash
jupyter lab regression.ipynb classification.ipynb clustering.ipynb
```

Download the datasets first — see [`../data/README.md`](../data/README.md). All three notebooks are deterministic (`random_state=42`) and run top to bottom with no manual steps. Paths are relative to this folder, so launch Jupyter with `notebooks/` as the working directory. Paths are relative to this folder, so launch Jupyter with `notebooks/` as the working directory.
