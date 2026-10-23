<div align="center">

<h1>
  <img src="assets/badges/hero.svg" alt="ChargeIQ — Battery Health, Vehicle Classification and Charging Behaviour Analytics" width="840">
</h1>

*Three independent Machine Learning tracks. Three angles on the EV ecosystem.*

<br>

[![Open the live app](https://img.shields.io/badge/▶%20Open%20the%20live%20app-chargeiq.streamlit.app-2E7D32?style=for-the-badge&logo=streamlit&logoColor=white)](https://chargeiq.streamlit.app)

[<img src="assets/badges/btn-regression.svg" alt="Jump to the Regression track" height="44">](#-track-1--regression--battery-state-of-health)
[<img src="assets/badges/btn-classification.svg" alt="Jump to the Classification track" height="44">](#-track-2--classification--electric-vehicle-type)
[<img src="assets/badges/btn-gallery.svg" alt="Jump to the Figure Gallery" height="44">](#figure-gallery)
[<img src="assets/badges/btn-progress.svg" alt="Open the Progress Board" height="44">](PROGRESS.md)

<br>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

[![Course](https://img.shields.io/badge/23CSE301-Machine%20Learning-0A66C2?style=flat-square)](#)
[![Review 2](https://img.shields.io/badge/Review%202-Implemented-2E7D32?style=flat-square)](PROGRESS.md)
[![Bonus](https://img.shields.io/badge/Bonus-GUI%20%2B%20public%20deployment-2E7D32?style=flat-square)](#the-live-app)

[<img src="assets/badges/badge-review1-complete.svg" alt="Review 1: Complete" height="28">](PROGRESS.md)
[<img src="assets/badges/badge-regression-10-10.svg" alt="Regression: 10 of 10 algorithms" height="28">](#-track-1--regression--battery-state-of-health)
[<img src="assets/badges/badge-classification-a-5-5.svg" alt="Classification Part A: 5 of 5" height="28">](#-track-2--classification--electric-vehicle-type)
[![Classification Part B: 5 of 5](https://img.shields.io/badge/Classification%20Part%20B-5%2F5-43A047?style=flat-square)](#classification-part-b)
[![Clustering: 2 of 2](https://img.shields.io/badge/Clustering-2%2F2-43A047?style=flat-square)](#clustering-track)
[<img src="assets/badges/badge-best-r2.svg" alt="Best R-squared: 0.815" height="28">](#-results--all-10-algorithms)
[<img src="assets/badges/badge-best-f1.svg" alt="Best weighted F1: 0.9999" height="28">](#classification-comparison)

</div>

---

## 💡 The Idea

An electric vehicle generates three completely different kinds of question. **ChargeIQ answers each with its own dataset, its own preprocessing and its own models** — the battery, the vehicle, and the driver.

| | Track | Real-world question | Target | Status |
|:-:|:--|:--|:--|:-:|
| 🔋 | **Regression** | *"How much life is left in this battery?"* | **State-of-Health (SOH, %)** — continuous | ✅ 10/10 |
| 🚗 | **Classification** | *"What kind of electric vehicle is this?"* | **`Electric Vehicle Type`** — BEV vs PHEV | ✅ 10/10 — Part A + Part B |
| 🔌 | **Clustering** | *"What charging-behaviour patterns exist among drivers?"* | *None — unsupervised* | ✅ 2/2 |

> [!IMPORTANT]
> The three tracks are **fully independent**. Each has its own dataset, its own preprocessing, its own target and its own evaluation. Nothing is shared between them — no joint feature matrix, no joint split, no joint model.

> [!NOTE]
> For the Regression track the **single target is SOH (%)**, computed as `SOH_pct = 100 × Capacity / NOMINAL_CAPACITY`. Remaining Useful Life is *not* modelled anywhere in this project.

---

<a id="the-live-app"></a>

## 🖥️ The Live App

**[chargeiq.streamlit.app](https://chargeiq.streamlit.app)** — the three trained models, behind an interactive story anyone can open in a browser. No sign-in, nothing to install.

| | Chapters | Try it yourself |
|:-:|:--|:--|
| 💡 | **1 · The problem** — the three questions | — |
| 🚗 | **2 · Meet the data · 3 · How the model thinks · 4 · Make a prediction** | Describe a vehicle → BEV or PHEV, with the exact path the decision tree took |
| 🔋 | **5 · Meet the data · 6 · How the model learns · 7 · Estimate battery health** | Pick a cycle from a battery the model never saw → predicted vs measured SOH, then adjust its measurements |
| 🔌 | **8 · Meet the data · 9 · How the model groups · 10 · Place a charging session** | Describe a charging session → its K-Means group and where it lands on the PCA map |
| 🏁 | **11 · Conclusion** — the three tracks side by side, with their limits | — |

**How it is built**

- **The notebooks' own models.** None of the notebooks saved its trained model, so `models/export_*.py` rebuilds each one by replaying the notebook's cells verbatim, and writes it only after the replay matches the results stored in the notebook — the classification confusion matrix `[[47509, 5], [0, 11325]]`, the regression R² 0.8153 and its five GroupKFold folds, and every one of the 100,000 stored K-Means labels. Each export is deterministic: run twice, it writes identical files.
- **Every claim is traceable.** Interpretive text is quoted from the notebooks and cites its cell; an automated test checks each quote word for word, and checks every figure against the notebook output it came from.
- **Tested and deployed.** 259 automated tests (`app/tests/`) cover the models, the app and deployment readiness. The app runs on Streamlit Community Cloud with Python 3.13 and exactly pinned dependencies ([`app/requirements.txt`](app/requirements.txt)); every push to `main` redeploys it. See [`app/DEPLOYMENT.md`](app/DEPLOYMENT.md).

> [!TIP]
> The app sleeps after 12 hours without visitors. If you see a wake-up screen, press the button and give it a moment.

---

## 🧭 The Pipelines

### 🔋 Regression Track

```mermaid
flowchart LR
    D1["📥 NASA Li-ion PCoE<br/>metadata.csv 7,565 × 10<br/>+ ~7.5k per-cycle CSVs"]
    P1["🧪 Preprocessing<br/>structural join · curve feature extraction<br/>drop corrupted cells · impute · cap"]
    T1["🎯 Target<br/>SOH_pct (%)"]
    S1["✂️ GroupShuffleSplit by battery_id<br/>StandardScaler (train-fit only)<br/>+ cycle_index_squared"]
    A1["🤖 10 Regressors<br/>Linear · Ridge · Lasso · ElasticNet · Polynomial<br/>DecisionTree · RandomForest · GradientBoosting · SVR · KNN"]
    E1["📊 Evaluation<br/>R² · RMSE · MAE · 5-fold GroupKFold CV<br/>residual · pred-vs-actual · feature importance"]
    D1 --> P1 --> T1 --> S1 --> A1 --> E1
    classDef done fill:#2E7D32,stroke:#1B5E20,color:#fff
    class D1,P1,T1,S1,A1,E1 done
```

### 🚗 Classification Track

```mermaid
flowchart LR
    D2["📥 WA EV Population Data<br/>294,193 × 16"]
    P2["🧪 Preprocessing<br/>median/mode impute · drop identifiers<br/>drop CAFV target-proxy · Vehicle Age"]
    T2["🎯 Target<br/>Electric Vehicle Type<br/>BEV 80.75% / PHEV 19.25%"]
    S2["✂️ Stratified 80:20 split<br/>60k training subset<br/>OneHotEncoder + StandardScaler (train-fit only)"]
    A2["🤖 10 Classifiers<br/>Part A: LogisticRegression · KNN · GaussianNB · DecisionTree · SVC<br/>Part B: RandomForest · AdaBoost · GradientBoosting · Bagging · MLP"]
    E2["📊 Evaluation<br/>Accuracy · Precision · Recall · weighted F1 · ROC-AUC<br/>confusion matrices · ROC curves · 5-fold CV"]
    D2 --> P2 --> T2 --> S2 --> A2 --> E2
    classDef done fill:#2E7D32,stroke:#1B5E20,color:#fff
    class D2,P2,T2,S2,A2,E2 done
```

### 🔌 Clustering Track

```mermaid
flowchart LR
    D3["📥 Synthetic EV charging sessions<br/>1,965,239 × 4"]
    S3["🎲 Reproducible sample<br/>100,000 sessions"]
    P3["🧪 Preprocessing<br/>median impute · Yeo-Johnson<br/>StandardScaler"]
    T3["🎯 No labels during fitting"]
    A3["🤖 K-Means (K = 3)<br/>Agglomerative — Ward + kNN connectivity"]
    E3["📊 Silhouette · Davies-Bouldin · Calinski-Harabasz<br/>Elbow · Dendrograms · PCA-2D · t-SNE"]
    D3 --> S3 --> P3 --> T3 --> A3 --> E3
    classDef done fill:#2E7D32,stroke:#1B5E20,color:#fff
    class D3,S3,P3,T3,A3,E3 done
```

---

## 🔋 Track 1 — Regression — Battery State of Health

**Notebook:** [`notebooks/regression.ipynb`](notebooks/regression.ipynb) · **Dataset:** NASA Li-ion Battery Aging (PCoE) · **Target:** `SOH_pct` (continuous, %)

<details open>
<summary><b>📦 Dataset</b></summary>

<br>

| | |
|:--|:--|
| **Source** | NASA Ames Prognostics Center of Excellence — cleaned Kaggle mirror `patrickfleith/nasa-battery-dataset` |
| **Structure** | `metadata.csv` (**7,565 × 10**) indexing ~7.5k raw per-cycle CSVs via its `filename` column |
| **Row types** | 2,815 charge · 2,794 discharge · 1,956 impedance |
| **Cells** | 34 batteries → **30** after dropping corrupted-log cells (B0049–B0052) → **2,694** usable discharge-cycle rows |
| **Target** | `SOH_pct` = `100 × Capacity / NOMINAL_CAPACITY` — range 19.8 % – 122.2 %, mean 68.1 % |
| **Local path** | `data/Regression_Dataset/` — see [Quickstart](#-quickstart) |

</details>

### Why SOH, and not raw Capacity?

Raw `Capacity` (Ah) is not comparable across cells with different nominal ratings. Normalising to **SOH % against nominal capacity** gives a target that is physically meaningful, comparable across every cell, and directly interpretable as *"how much of its original life does this battery still have?"*

### 🧠 The methodological core — *split by battery, not by row*

> [!NOTE]
> This is the most important design decision in the track.

This is **panel data** — repeated cycles from the same ~30 batteries, not independent rows. Under a naive random row-level split, cycles from the *same* battery land in both train and test, and the model can partly fit battery-specific quirks through the `battery_id` one-hot column instead of learning the general physics. The scores look better; the question being answered is easier and less useful.

**The fix, applied throughout:**

- 🔒 **`GroupShuffleSplit` grouped on `battery_id`** — every cycle from a battery goes entirely into train *or* test, never both.
- 🚫 **`battery_id` dropped as a model feature** — a battery-level split makes it structurally unable to generalise to unseen cells.
- 📏 **`StandardScaler` fitted on the training split only.**
- 🔁 **`GroupKFold`** used for cross-validation and for polynomial degree selection, so grouping is respected there too.
- 🎲 **`random_state=42`** everywhere.

| | Batteries | Rows |
|:--|--:|--:|
| **Train** | 24 | 2,345 |
| **Test** *(fully unseen cells)* | 6 | 349 |
| **Overlap** | 0 | 0 |

### 🧬 Feature set — 10 features

| # | Feature | Origin | Intuition |
|:-:|:--|:--|:--|
| 1 | `ambient_temperature` | metadata | Test-protocol temperature group |
| 2 | `Re` | impedance rows | Electrolyte resistance — rises as the cell ages |
| 3 | `Rct` | impedance rows | Charge-transfer resistance — rises as the cell ages |
| 4 | `cycle_index` | derived | How many cycles this cell has been through |
| 5 | `early_voltage_sag_rate` | 🧪 raw discharge curve | Voltage drop in the first 300 s — degraded cells sag faster |
| 6 | `early_temp_rise_rate` | 🧪 raw discharge curve | Early self-heating rate |
| 7 | `charge_duration` | 🧪 raw charge curve | Total time to recharge |
| 8 | `charge_max_temp` | 🧪 raw charge curve | Peak temperature while charging |
| 9 | `charge_time_to_cv` | 🧪 raw charge curve | Time in constant-current before constant-voltage |
| 10 | `cycle_index_squared` | ⚙️ **engineered** | SOH fades in a **curve**, not a line |

> 💡 **Why only the *early* part of the discharge curve?** Capacity *is defined as* discharge current integrated over full discharge time — verified on one file: `0.995 A × (6179 s / 3600) ≈ 1.71 Ah` against that row's recorded `Capacity = 1.674 Ah`. A feature built from the full curve would restate the target rather than predict it. Restricting to the first 300 s keeps the physics and removes the leakage.

### 📈 Results — all 10 algorithms

Every model trained on the **same** preprocessed split and evaluated on the **same** held-out test set of 6 unseen batteries. RMSE and MAE are in SOH percentage points.

| Rank | Algorithm | Best parameters | R² ⬆ | RMSE ⬇ | MAE ⬇ |
|:-:|:--|:--|--:|--:|--:|
| 🥇 | **Gradient Boosting Regressor** | `learning_rate=0.1, max_depth=3, n_estimators=200` | **0.8153** | **8.0449** | 5.6427 |
| 🥈 | K-Nearest Neighbors Regressor | `n_neighbors=7, p=1, weights=distance` | 0.8013 | 8.3445 | **5.1617** |
| 🥉 | Random Forest Regressor | `max_depth=None, min_samples_leaf=4, n_estimators=200` | 0.7924 | 8.5277 | 5.5031 |
| 4 | Support Vector Regressor | `C=100, epsilon=0.1, kernel=rbf` | 0.7919 | 8.5396 | 6.5133 |
| 5 | Polynomial Regression | `degree=2` *(GroupKFold-selected)* | 0.7862 | 8.6539 | 5.9320 |
| 6 | Lasso Regression | `alpha=0.5` | 0.6820 | 10.5553 | 8.6171 |
| 7 | Linear Regression | *baseline* | 0.6671 | 10.7999 | 8.3770 |
| 8 | Ridge Regression | `alpha=100` | 0.6650 | 10.8331 | 8.8837 |
| 9 | ElasticNet Regression | `alpha=0.1, l1_ratio=0.1` | 0.6558 | 10.9816 | 9.1909 |
| 10 | Decision Tree Regressor | `max_depth=3, min_samples_leaf=1, min_samples_split=2` | 0.6525 | 11.0345 | 8.3325 |

**Best model: Gradient Boosting** — R² 0.8153, within ±5.64 SOH-points of mean absolute error on batteries it has never seen.

### 🔁 5-fold cross-validation — two best models

`GroupKFold(n_splits=5)` on `battery_id`, so folds never split a battery.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | **Std** |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | 0.8998 | 0.9209 | 0.6566 | 0.8654 | 0.7764 | **0.8238** | 0.0971 |
| KNN | 0.8242 | 0.6071 | 0.7442 | 0.6381 | 0.6897 | **0.7006** | 0.0774 |

Gradient Boosting leads on mean CV R² by a clear margin (0.8238 vs 0.7006), and its cross-validated mean sits close to its test-set R² of 0.8153 — the held-out result is not a lucky split. Its fold-to-fold spread is slightly wider than KNN's (σ 0.097 vs 0.077), driven almost entirely by fold 3 (0.6566); batteries differ enough that any single fold can be hard.

### 🎛️ Hyperparameter tuning

`GridSearchCV` (5-fold) applied to **8 models** — Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, SVR and KNN. Polynomial **degree** is selected separately, via `GroupKFold` + `cross_val_score` on the training data only, so the held-out set plays no part in choosing it. Before/after on the held-out test set:

| Model | Original R² | Tuned R² | Δ |
|:--|--:|--:|--:|
| Gradient Boosting | 0.8053 | 0.8153 | **+0.0100** |
| Decision Tree | 0.6672 | 0.6525 | −0.0148 |

Tuning improved one model and not the other — a genuine finding, kept in the notebook rather than hidden.

<details>
<summary><b>🔍 Two findings worth reading</b></summary>

<br>

**1 · Collinearity makes the linear coefficients lie.** In the Linear baseline, `cycle_index` and `cycle_index_squared` have the *smallest* coefficients (−0.41 and 0.81), while `charge_time_to_cv` (15.35), `charge_max_temp` (−8.59) and `ambient_temperature` (7.42) dominate. That is not "cycle count doesn't matter" — `ambient_temperature` correlates **0.94** with `charge_max_temp`, **0.82** with `charge_time_to_cv` and **−0.78** with `Rct`, because NASA's protocol fixes a matching charge/discharge regime per temperature group. Those features largely stand in for *which temperature group a cell belongs to*, and the collinearity even flips coefficient signs relative to each feature's own simple correlation with SOH.

**2 · Polynomial degree must be validated, not assumed.** Degree selection uses `GroupKFold` CV on the **training data only** — the held-out test set plays no part in choosing the degree. Degree 2 (54 terms) wins; degree 3 explodes to 219 terms and collapses under its own variance.

</details>

---

## 🚗 Track 2 — Classification — Electric Vehicle Type

**Notebook:** [`notebooks/classification.ipynb`](notebooks/classification.ipynb) · **Dataset:** Washington State EV Population Data · **Target:** `Electric Vehicle Type` (binary)

> [!IMPORTANT]
> This track uses a **completely separate dataset** from the Regression track. There is no shared data, no shared preprocessing and no shared split between the two.

<details open>
<summary><b>📦 Dataset</b></summary>

<br>

| | |
|:--|:--|
| **Source** | [data.wa.gov — Electric Vehicle Population Data](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2/about_data) (official) |
| **Shape** | **294,193 × 16** |
| **Target** | **`Electric Vehicle Type`** — 2 classes: Battery Electric Vehicle (BEV) **80.75 %** · Plug-in Hybrid Electric Vehicle (PHEV) **19.25 %** |
| **Local path** | `data/Classification_Dataset/Electric_Vehicle_Population_Data.csv` |

</details>

### 🧪 Preprocessing

1. **Audit** — shape, dtypes, missing values, duplicates, target distribution.
2. **Cleaning** — median imputation for numerical columns, mode imputation for categorical; duplicates removed. Numerical outliers identified by IQR but **retained**, since extreme vehicle specs are usually genuine rather than data-entry errors.
3. **Column removal** — three columns dropped, for two different reasons:
   - `VIN (1-10)`, `DOL Vehicle ID` — identifiers, they label a record rather than describe a vehicle.
   - `Clean Alternative Fuel Vehicle (CAFV) Eligibility` — **dropped as a target proxy.** CAFV eligibility is derived from vehicle type and electric range, so leaving it in would leak the answer.
4. **Feature engineering** — `Vehicle Age = 2026 − Model Year`, giving the models an explicit age measure instead of a raw year.
5. **Split** — stratified 80:20 (train 235,354 / test 58,839), then a stratified **60,000-row training subset** so the distance- and kernel-based models remain tractable. Class proportions preserved exactly at every step.
6. **Encoding & scaling** — `ColumnTransformer` with `OneHotEncoder` for categoricals and `StandardScaler` for numericals, **fitted on training data only**.

### 📊 Results — all 5 Part-A algorithms

All five trained on the same 60,000-row preprocessed training subset and evaluated on the same held-out 58,839-row test set. Precision, recall and F1 are **weighted**, appropriate for the 81/19 imbalance. A confusion matrix is plotted for every model.

| Rank | Algorithm | Tuning | Accuracy | Precision | Recall | Weighted F1 |
|:-:|:--|:--|--:|--:|--:|--:|
| 🥇 | **Decision Tree Classifier** | `max_depth=7` *(validation-tuned)* | **0.9999** | **0.9999** | **0.9999** | **0.9999** |
| 🥈 | Support Vector Machine (SVC) | `kernel=rbf, C=1.0` *(grid-tuned)* | 0.9975 | 0.9975 | 0.9975 | 0.9975 |
| 🥉 | Logistic Regression | `max_iter=1000` *(baseline)* | 0.9933 | 0.9933 | 0.9933 | 0.9933 |
| 4 | Gaussian Naive Bayes | — *(15,000-row stratified subset)* | 0.9834 | 0.9841 | 0.9834 | 0.9835 |
| 5 | K-Nearest Neighbors | `k=3`, Euclidean *(validation-tuned)* | 0.9725 | 0.9722 | 0.9725 | 0.9723 |

**Best model in Part A: Decision Tree** on both accuracy and weighted F1.

<details>
<summary><b>🔬 Model-specific notes</b></summary>

<br>

- **Logistic Regression** — coefficients and odds ratios reported for the top 10 positive and negative encoded features.
- **KNN** — `k` swept over `[3, 5, 7, 9, 11]` on a stratified 30,000-row tuning subset with a held-out validation split; Euclidean distance used throughout, appropriate because all numerical features are standardised and categoricals are one-hot encoded.
- **Gaussian Naive Bayes** — requires dense input, so it is trained on a stratified 25 % subset of the training rows (15,000), densified, and evaluated on the full test set in batches; the conditional-independence assumption is discussed in the notebook.
- **Decision Tree** — `max_depth` swept over `[3, 5, 7, 10, 15, 20]` on a validation split carved from the 60k training subset; the fitted tree is visualised to 3 levels.
- **SVM** — `C ∈ [0.1, 1, 10]` × `kernel ∈ [linear, rbf]` tuned on a stratified 10,000-row subset, since full-size SVM training is expensive; the final model is refit on all 60,000 rows.

</details>

<a id="classification-part-b"></a>

### 🧩 Part B — five more classifiers

The same 60,000-row training subset, the same 58,839-row test set and the same metrics as Part A, now with **ROC-AUC** for every model.

| # | Algorithm | Configuration | Accuracy | Weighted F1 | ROC-AUC |
|:-:|:--|:--|--:|--:|--:|
| 6 | Random Forest | `n_estimators=100, max_depth=None` — tuned over `{100, 200} × {10, 20, None}` on a 20,000-row subset | 0.9988 | 0.9988 | 1.0000 |
| 7 | AdaBoost | depth-1 stumps; `n_estimators=50, learning_rate=0.5` — tuned over `{50, 100} × {0.5, 1.0}` | 0.9994 | 0.9994 | 1.0000 |
| 8 | Gradient Boosting | `n_estimators=100, learning_rate=0.1, max_depth=3` on a 100-component `TruncatedSVD` of the 1,694 encoded features | 0.9928 | 0.9928 | 0.9995 |
| 9 | Bagging | 50 decision trees (`max_depth=15`), `max_samples=0.8` | **0.9999** | **0.9999** | 0.9999 |
| 10 | MLP | `hidden_layer_sizes=(100, 50)`, ReLU, Adam, early stopping — stopped after 20 iterations | 0.9978 | 0.9978 | 1.0000 |

Part B also adds a confusion matrix and a ROC curve for every model, and Random Forest feature importances (`classification.ipynb` §6–10).

<a id="classification-comparison"></a>

### 🏆 All 10 classifiers compared

| Rank | Model | Part | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|:-:|:--|:-:|--:|--:|--:|--:|--:|
| 🥇 | **Decision Tree** | A | **0.9999** | **0.9999** | **0.9999** | **0.9999** | 0.9999 |
| 🥈 | Bagging | B | **0.9999** | **0.9999** | **0.9999** | **0.9999** | 0.9999 |
| 🥉 | AdaBoost | B | 0.9994 | 0.9994 | 0.9994 | 0.9994 | **1.0000** |
| 4 | Random Forest | B | 0.9988 | 0.9988 | 0.9988 | 0.9988 | **1.0000** |
| 5 | MLP | B | 0.9978 | 0.9978 | 0.9978 | 0.9978 | **1.0000** |
| 6 | SVM | A | 0.9975 | 0.9975 | 0.9975 | 0.9975 | 0.9998 |
| 7 | Logistic Regression | A | 0.9933 | 0.9933 | 0.9933 | 0.9933 | 0.9997 |
| 8 | Gradient Boosting | B | 0.9928 | 0.9928 | 0.9928 | 0.9928 | 0.9995 |
| 9 | Gaussian Naive Bayes | A | 0.9834 | 0.9841 | 0.9834 | 0.9835 | 0.9840 |
| 10 | KNN | A | 0.9725 | 0.9722 | 0.9725 | 0.9723 | 0.9837 |

*Ranked as in the notebook (cell 81). Every ROC-AUC shown as 1.0000 is rounded to four places.*

**5-fold cross-validation** of the two best models, on the 60,000-row training subset:

| Model | Mean weighted F1 | Std |
|:--|--:|--:|
| Decision Tree | **0.9998** | 0.0004 |
| Bagging | 0.9995 | 0.0004 |

**Final model: Decision Tree** (`max_depth=7`) — best on weighted F1 and accuracy, and slightly ahead of Bagging in cross-validation; AdaBoost has the highest ROC-AUC after rounding. The selected tree has 15 leaves, and its first split is `Electric Range ≤ 0.5`.

> [!IMPORTANT]
> **Why the scores are so high.** In this dataset an `Electric Range` of 0 does not mean "no electric range": all 192,477 registrations with a range of 0 are BEVs, and each carries the CAFV value *"Eligibility unknown as battery range has not been researched"*. A range of 0 therefore carries the same information as the CAFV column that was dropped as a target proxy — and it is exactly the tree's first rule. `Make` and `Model` are near-deterministic as well. The notebook's conclusion (cell 87) calls for this leakage audit; [Chapter 3 of the app](https://chargeiq.streamlit.app/vehicle-model) shows the tree's actual rules.

---

<a id="clustering-track"></a>

## 🔌 Track 3 — Clustering — Charging Sessions

**Notebook:** [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) · **Dataset:** Electric Vehicle Charging Dataset *(synthetic)* · **Target:** none — unsupervised

> [!IMPORTANT]
> A third, completely separate dataset. And it is **synthetic**: the sessions were generated, not recorded, so the groups describe patterns in generated data.

<details open>
<summary><b>📦 Dataset</b></summary>

<br>

| | |
|:--|:--|
| **Source** | Gholizadeh, N. (2024). *Electric Vehicle Charging Dataset*. Mendeley Data, V1. [doi:10.17632/5zrtmp7gwd.1](https://doi.org/10.17632/5zrtmp7gwd.1) — via its [Kaggle mirror](https://www.kaggle.com/datasets/mexwell/electric-vehicle-charging-dataset). Licence **CC BY 4.0**. |
| **How it was made** | Generated with conditional tabular GANs (CTGAN) and kernel density estimation from real sessions at a Caltech EV parking lot — 29,600 synthetic days from 185 real ones |
| **Shape** | **1,965,239 × 4** — `connectionTime_decimal` (hour of day, 0–24), `chargingDuration` (hours), `kWhDelivered` (kWh), `dayIndicator` (day number, not used) |
| **Medians** | connection 16:10 · duration 2.55 h · energy 9.2 kWh |
| **Local path** | `data/Clustering_Dataset/SYNTHETIC_EV_DATA.csv` (120 MB, gitignored) |

</details>

### 🧪 Preprocessing

1. **Audit and cleaning** — no missing values and no duplicates; all 1,965,239 sessions retained. Outliers are kept, since they may be genuine sessions.
2. **Sample** — a reproducible **100,000-session** sample (`random_state=42`) for fitting and evaluation. The exact sample, with both algorithms' labels, is saved as [`notebooks/EV_clustering_sample_with_labels.csv`](notebooks/EV_clustering_sample_with_labels.csv).
3. **Features** — `connectionTime_decimal`, `chargingDuration`, `kWhDelivered`; `dayIndicator` is left out.
4. **Transform** — median imputation → **Yeo–Johnson** power transform (to reduce skew) → `StandardScaler`.
5. **Choosing K** — the elbow curve over K = 2…10; **K = 3** is used for both algorithms, so they can be compared on equal terms.

### 📊 Results — both algorithms

Silhouette is estimated on a shared 5,000-session subset; Davies–Bouldin and Calinski–Harabasz use all 100,000 sessions.

| Algorithm | Silhouette ⬆ | Davies–Bouldin ⬇ | Calinski–Harabasz ⬆ |
|:--|--:|--:|--:|
| **K-Means** (`n_init=5, max_iter=100`) | **0.4094** | **0.9195** | **84,854.2** |
| Agglomerative (Ward, 10-nearest-neighbour connectivity) | 0.3898 | 0.9767 | 73,872.0 |

**K-Means ranks first on all three metrics.** These are internal metrics — the data has no ground-truth labels to score against. K-Means is also the only one of the two that can place a *new* session in a group, which is why it powers the app.

**The three K-Means groups** — each group's mean session *(group numbers are arbitrary labels)*:

| Group | Sessions | Connection time | Duration | Energy |
|:-:|--:|:-:|--:|--:|
| 0 | 35,099 (35 %) | 17:50 | 1.92 h | 6.1 kWh |
| 1 | 46,195 (46 %) | 16:43 | 6.26 h | 13.9 kWh |
| 2 | 18,706 (19 %) | 02:43 | 1.31 h | 4.8 kWh |

<details>
<summary><b>Agglomerative groups, for comparison</b></summary>

<br>

| Group | Sessions | Connection time | Duration | Energy |
|:-:|--:|:-:|--:|--:|
| 0 | 49,776 (50 %) | 17:23 | 5.61 h | 13.7 kWh |
| 1 | 18,803 (19 %) | 02:16 | 1.79 h | 5.4 kWh |
| 2 | 31,421 (31 %) | 17:15 | 2.16 h | 5.1 kWh |

*Its group numbers are independent of K-Means'.*

</details>

The first two principal components keep **88.53 %** of the variance; the notebook plots both algorithms' groups on that plane, a t-SNE view of 1,000 sessions, and dendrograms of 500 sessions for single, complete, average and Ward linkage.

---

<a id="figure-gallery"></a>

## 🖼️ Figure Gallery

All **35** figures live in [`assets/figures/`](assets/figures). The regression notebook and Part A of the classification notebook write theirs with a `plt.savefig` call in the cell that draws them, so **Run All** regenerates them. The clustering notebook only displays its plots, so its 8 figures were extracted byte for byte from the notebook's stored outputs by [`models/export_clustering.py`](models/export_clustering.py). Classification Part B's figures (confusion matrices, ROC curves, Random Forest importances, the 10-model comparison) are shown inside `classification.ipynb` only.

### 🔋 Regression — EDA *(`regression.ipynb` §3)*

<div align="center">
<table>
<tr>
<td width="33%" align="center"><b>Raw-curve feature distributions</b></td>
<td width="33%" align="center"><b>Correlation heatmap</b></td>
<td width="33%" align="center"><b>Capacity fade & feature–target scatter</b></td>
</tr>
<tr>
<td><img src="assets/figures/regression/eda/01_raw_feature_distributions.png" alt="Distributions of the five raw-curve-derived features" width="100%"></td>
<td><img src="assets/figures/regression/eda/02_correlation_heatmap.png" alt="Correlation heatmap across all features and the target" width="100%"></td>
<td><img src="assets/figures/regression/eda/03_capacity_fade_and_scatter.png" alt="Capacity fade per battery and feature-target scatter plots" width="100%"></td>
</tr>
</table>
</div>

### 🔋 Regression — Models & Diagnostics *(`regression.ipynb` §11 & §14)*

The three **C4 rubric deliverables**. Feature importance comes from the Decision Tree (algorithm 6); the two diagnostic plots are for the best model, **Gradient Boosting**.

<div align="center">
<table>
<tr>
<td width="33%" align="center"><b>Decision Tree feature importance</b></td>
<td width="33%" align="center"><b>Predicted vs. actual SOH</b></td>
<td width="33%" align="center"><b>Residual plot</b></td>
</tr>
<tr>
<td><img src="assets/figures/regression/models/01_decision_tree_feature_importance.png" alt="Decision Tree feature importance across the ten engineered features" width="100%"></td>
<td><img src="assets/figures/regression/models/02_predicted_vs_actual.png" alt="Predicted versus actual SOH for the best model, with the perfect-prediction diagonal" width="100%"></td>
<td><img src="assets/figures/regression/models/03_residual_plot.png" alt="Residuals versus predicted SOH for the best model" width="100%"></td>
</tr>
</table>
</div>

### 🚗 Classification — EDA *(`classification.ipynb` §4)*

<div align="center">
<table>
<tr>
<td width="25%" align="center"><b>Numeric feature distributions</b></td>
<td width="25%" align="center"><b>Target distribution</b></td>
<td width="25%" align="center"><b>Correlation heatmap</b></td>
<td width="25%" align="center"><b>Electric Range vs. target</b></td>
</tr>
<tr>
<td><img src="assets/figures/classification/eda/01_numeric_feature_distributions.png" alt="Histograms of the five numerical vehicle features" width="100%"></td>
<td><img src="assets/figures/classification/eda/03_target_distribution.png" alt="Count of BEV versus PHEV vehicles" width="100%"></td>
<td><img src="assets/figures/classification/eda/04_correlation_heatmap.png" alt="Correlation heatmap of the numerical features" width="100%"></td>
<td><img src="assets/figures/classification/eda/05_electric_range_vs_target.png" alt="Electric range against vehicle type, coloured by class" width="100%"></td>
</tr>
</table>
</div>

### 🚗 Classification — Models *(`classification.ipynb` §9–14)*

<div align="center">
<table>
<tr>
<td width="25%" align="center"><b>All 5 models compared</b></td>
<td width="25%" align="center"><b>Decision Tree — confusion matrix</b></td>
<td width="25%" align="center"><b>Decision Tree — first 3 levels</b></td>
<td width="25%" align="center"><b>KNN — choosing <i>k</i></b></td>
</tr>
<tr>
<td><img src="assets/figures/classification/models/08_model_comparison.png" alt="Bar chart comparing accuracy, precision, recall and weighted F1 across the five classifiers" width="100%"></td>
<td><img src="assets/figures/classification/models/05_decision_tree_confusion_matrix.png" alt="Confusion matrix for the best classifier, the Decision Tree" width="100%"></td>
<td><img src="assets/figures/classification/models/06_decision_tree_visualisation.png" alt="Decision tree structure to a depth of three levels" width="100%"></td>
<td><img src="assets/figures/classification/models/02_knn_k_tuning.png" alt="Accuracy and weighted F1 across candidate values of k" width="100%"></td>
</tr>
</table>
</div>

### 🔌 Clustering *(`clustering.ipynb`)*

<div align="center">
<table>
<tr>
<td width="33%" align="center"><b>Feature distributions</b></td>
<td width="33%" align="center"><b>K-Means elbow curve</b></td>
<td width="33%" align="center"><b>Ward dendrogram</b></td>
</tr>
<tr>
<td><img src="assets/figures/clustering/eda/01_feature_distributions.png" alt="Histograms of connection time, charging duration and energy delivered" width="100%"></td>
<td><img src="assets/figures/clustering/models/01_kmeans_elbow_curve.png" alt="K-Means inertia against the number of clusters, K = 2 to 10" width="100%"></td>
<td><img src="assets/figures/clustering/models/03_ward_dendrogram.png" alt="Ward-linkage dendrogram of up to 500 sessions" width="100%"></td>
</tr>
<tr>
<td width="33%" align="center"><b>K-Means — PCA 2D</b></td>
<td width="33%" align="center"><b>Agglomerative — PCA 2D</b></td>
<td width="33%" align="center"><b>K-Means — t-SNE</b></td>
</tr>
<tr>
<td><img src="assets/figures/clustering/models/04_kmeans_pca.png" alt="K-Means clusters on the first two principal components" width="100%"></td>
<td><img src="assets/figures/clustering/models/05_agglomerative_pca.png" alt="Agglomerative clusters on the first two principal components" width="100%"></td>
<td><img src="assets/figures/clustering/models/06_kmeans_tsne.png" alt="K-Means clusters in a t-SNE projection of 1,000 sessions" width="100%"></td>
</tr>
</table>
</div>

<details>
<summary><b>🗂️ The remaining 15 figures</b></summary>

<br>

**Classification — the other four confusion matrices**

<div align="center">
<table>
<tr>
<td width="25%" align="center"><b>Logistic Regression</b></td>
<td width="25%" align="center"><b>KNN (k=3)</b></td>
<td width="25%" align="center"><b>Gaussian Naive Bayes</b><br><sub>earlier run — the current cell no longer saves it</sub></td>
<td width="25%" align="center"><b>SVM (rbf, C=1.0)</b></td>
</tr>
<tr>
<td><img src="assets/figures/classification/models/01_logistic_regression_confusion_matrix.png" alt="Logistic Regression confusion matrix" width="100%"></td>
<td><img src="assets/figures/classification/models/03_knn_confusion_matrix.png" alt="KNN confusion matrix" width="100%"></td>
<td><img src="assets/figures/classification/models/04_naive_bayes_confusion_matrix.png" alt="Gaussian Naive Bayes confusion matrix" width="100%"></td>
<td><img src="assets/figures/classification/models/07_svm_confusion_matrix.png" alt="Support Vector Machine confusion matrix" width="100%"></td>
</tr>
</table>
</div>

**Clustering — correlation heatmap and linkage comparison**

<div align="center">
<table>
<tr>
<td width="40%" align="center"><b>Correlation heatmap</b></td>
<td width="60%" align="center"><b>Single · complete · average linkage</b></td>
</tr>
<tr>
<td><img src="assets/figures/clustering/eda/02_correlation_heatmap.png" alt="Correlation heatmap of the three charging features" width="100%"></td>
<td><img src="assets/figures/clustering/models/02_linkage_comparison_dendrograms.png" alt="Dendrograms for single, complete and average linkage" width="100%"></td>
</tr>
</table>
</div>

**Classification — Model Year vs. target**

<div align="center">
<img src="assets/figures/classification/eda/06_model_year_vs_target.png" alt="Model year against vehicle type, coloured by class" width="50%">
</div>

**Classification — top-10 category counts** *(one per categorical feature)*

<div align="center">
<table>
<tr>
<td width="25%" align="center"><b>County</b></td>
<td width="25%" align="center"><b>City</b></td>
<td width="25%" align="center"><b>State</b></td>
<td width="25%" align="center"><b>Make</b></td>
</tr>
<tr>
<td><img src="assets/figures/classification/eda/02_top_categories_county.png" alt="Top 10 counties by vehicle count" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_city.png" alt="Top 10 cities by vehicle count" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_state.png" alt="Top 10 states by vehicle count" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_make.png" alt="Top 10 makes by vehicle count" width="100%"></td>
</tr>
<tr>
<td width="25%" align="center"><b>Model</b></td>
<td width="25%" align="center"><b>Electric Vehicle Type</b></td>
<td width="25%" align="center"><b>CAFV Eligibility</b></td>
<td width="25%" align="center"><b>Electric Utility</b></td>
</tr>
<tr>
<td><img src="assets/figures/classification/eda/02_top_categories_model.png" alt="Top 10 models by vehicle count" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_electric_vehicle_type.png" alt="Vehicle counts by electric vehicle type" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_clean_alternative_fuel_vehicle_cafv_eligibility.png" alt="Vehicle counts by CAFV eligibility category" width="100%"></td>
<td><img src="assets/figures/classification/eda/02_top_categories_electric_utility.png" alt="Top 10 electric utilities by vehicle count" width="100%"></td>
</tr>
</table>
</div>

</details>

---

## 🗂️ Repository Structure

```
ChargeIQ/
├── README.md                    ← you are here
├── PROGRESS.md                  ← live status board vs. the rubric
├── requirements.txt             ← pinned dependencies for the notebooks
├── .streamlit/config.toml       ← the app's theme (Streamlit reads it from the repo root)
│
├── data/                        ← datasets (downloaded, not committed)
│   ├── README.md                   sources, layout, column descriptions
│   ├── Regression_Dataset/      ← 🔋 regression track only
│   │   ├── metadata.csv            7,565 × 10 — indexes every raw cycle
│   │   ├── data/                   ~7.5k per-cycle V/I/T curve CSVs
│   │   └── extra_infos/            NASA per-battery-group README files
│   ├── Classification_Dataset/  ← 🚗 classification track only
│   │   └── Electric_Vehicle_Population_Data.csv    294,193 × 16
│   └── Clustering_Dataset/      ← 🔌 clustering track only
│       └── SYNTHETIC_EV_DATA.csv   1,965,239 × 4
│
├── notebooks/
│   ├── regression.ipynb         ✅ SOH — sections 0 → 15, all 10 algorithms
│   ├── classification.ipynb     ✅ Electric Vehicle Type — Part A + Part B, 10 algorithms
│   ├── clustering.ipynb         ✅ charging sessions — K-Means + Agglomerative
│   ├── EV_clustering_sample_with_labels.csv    the 100,000-session sample, with labels
│   └── EV_clustering_evaluation_metrics.csv    both algorithms' metrics
│
├── models/                      ← the three trained models, rebuilt from the notebooks
│   ├── export_classification.py · export_regression.py · export_clustering.py
│   └── <track>_model.joblib + <track>_metadata.json, one pair per track
│
├── app/                         ← 🖥️ the live Streamlit app (bonus track)
│   ├── streamlit_app.py            entry point
│   ├── chapters/                   the 11 chapters of the story
│   ├── core/                       inference, one module per track — no Streamlit
│   ├── content/                    the notebooks' words, each quote cited by cell
│   ├── ui/                         navigation, components, styling
│   ├── tests/                      259 automated tests
│   ├── requirements.txt            the app's exact pins (used by Streamlit Cloud)
│   └── DEPLOYMENT.md               how to deploy, verify and troubleshoot
│
└── assets/figures/              ← 35 figures
    ├── regression/              eda/ 3 · models/ 3
    ├── classification/          eda/ 13 · models/ 8
    └── clustering/              eda/ 2 · models/ 6
```

---

## 🚀 Quickstart

**1 · Clone the repository**

```bash
git clone https://github.com/Roots26Hz/ChargeIQ.git
```

**2 · Create and activate a virtual environment** *(Python 3.13)*

```bash
python3 -m venv .venv && source .venv/bin/activate
```

**3 · Install dependencies**

```bash
pip install -r requirements.txt
```

**4 · Download the datasets** using the links in [`data/README.md`](data/README.md), into the layout shown above. All three are gitignored and must be fetched separately.

**5 · (Optional) Point the regression notebook at a different folder.** It defaults to the repo-relative path and needs no edits if you followed step 4. To use a dataset stored elsewhere, set an environment variable instead of editing any cell:

```bash
export CHARGEIQ_REGRESSION_DATA="/path/to/your/cleaned_dataset"
```

**6 · Launch the notebooks**

```bash
jupyter lab notebooks/regression.ipynb notebooks/classification.ipynb notebooks/clustering.ipynb
```

Then **Run All** on each. All three are deterministic (`random_state=42`).

**7 · Run the app locally** *(optional — it is also [live](https://chargeiq.streamlit.app))*

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
python -m pytest app/tests -q        # the app's 259 tests
```

The app needs none of the datasets: it reads only `models/` and `assets/figures/`. To rebuild a model from its notebook, run `python models/export_<track>.py` — it refuses to write unless the replay matches the notebook (the regression export needs the local NASA data).

---

## 📐 Rubric Coverage Map

Where each rubric criterion lives. Full mark-by-mark tracking is in **[PROGRESS.md](PROGRESS.md)**.

**Review 1**

| § | Criterion | Where | Status |
|:-:|:--|:--|:-:|
| A1 | Dataset loading & audit | `regression.ipynb` §1 · `classification.ipynb` §3 | ✅ |
| A2 | EDA visualisations | `regression.ipynb` §3 · `classification.ipynb` §4.1–4.4 — see [Figure Gallery](#figure-gallery) | ✅ |
| A3 | Insight commentary | Markdown observation after every major figure in both notebooks | ✅ |
| B1 | Data cleaning | `regression.ipynb` §4 · `classification.ipynb` §5 | ✅ |
| B2 | Encoding, scaling & splitting | `regression.ipynb` §6–7 · `classification.ipynb` §7–8 | ✅ |
| B3 | Feature engineering | `cycle_index_squared` · `Vehicle Age` | ✅ |
| C1 | All 10 regression algorithms | `regression.ipynb` §9 + §11 | ✅ 10/10 |
| C2 | Comparative evaluation table | `regression.ipynb` §12 — ranked by R² | ✅ |
| C3 | Hyperparameter tuning ≥ 2 models | GridSearchCV on 8 models (+ polynomial degree via GroupKFold CV); GB and DT deltas reported | ✅ |
| C4 | Residual / pred-vs-actual / feature-importance plots | `regression.ipynb` §14 + DT importance in §11 — all three in [`assets/figures/regression/models/`](assets/figures/regression/models) | ✅ |
| D1 | Classification Part A — 5 algorithms | `classification.ipynb` §9–13 | ✅ 5/5 |
| D2 | Part A evaluation + comparison table | `classification.ipynb` §14 | ✅ |

**Review 2**

| § | Criterion | Where | Status |
|:-:|:--|:--|:-:|
| A1 | Classification Part B — 5 algorithms | `classification.ipynb` §6–10 — Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP | ✅ 5/5 |
| A2 | Full 10-algorithm comparison, incl. ROC-AUC | `classification.ipynb` cell 81 — see [All 10 classifiers compared](#classification-comparison) | ✅ |
| A3 | Final model selection + tuning | Tuning: RF and AdaBoost grids, Decision Tree depth sweep; 5-fold CV (cell 84); selection (cells 86–87) | ✅ |
| B1 | Clustering — both algorithms | `clustering.ipynb` — K-Means (cell 15), Agglomerative (cell 21) | ✅ 2/2 |
| B2 | Silhouette · DB · CH + Elbow + Dendrogram | Metrics (cells 23, 33), elbow (cell 13), dendrograms (cells 17, 19) | ✅ |
| B3 | PCA-2D per algorithm + t-SNE | Cells 25, 27 (PCA) and 29 (t-SNE) | ✅ |
| C1 | Code quality | All three notebooks deterministic, with repo-relative dataset paths; the app is tested (259 tests) | ✅ |
| C2 | README & documentation | This README, [PROGRESS.md](PROGRESS.md), folder READMEs, [`app/DEPLOYMENT.md`](app/DEPLOYMENT.md) | ✅ |
| C3 | GitHub history + `requirements.txt` | Granular commit history; pinned [`requirements.txt`](requirements.txt) and [`app/requirements.txt`](app/requirements.txt) | ✅ |
| ⭐ | Bonus — GUI + public deployment | [chargeiq.streamlit.app](https://chargeiq.streamlit.app) — see [The Live App](#the-live-app) | ✅ +2 |

---

## 👥 Team

| Member | Roll No. | GitHub | Track focus |
|:--|:--|:--|:--|
| `Rutav Desai` | `CB.SC.U4CSE24664` | [@Roots26Hz](https://github.com/Roots26Hz) | `Regression (1-5)` · `Live app` |
| `Chikeersha Vedium` | `CB.SC.U4CSE24615` | [@chikeersha24-bot](https://github.com/chikeersha24-bot) | `Classification Part A + Part B` |
| `Harshitha Nityanand` | `CB.SC.U4CSE24624` | [@Harshi06-code](https://github.com/Harshi06-code) | `Regression (6-10)` · `Clustering` |

---

## 🤖 AI Assistance Disclosure

In line with §7 of the course guidelines: **Claude (Anthropic, via Claude Code)** was used for **code scaffolding and documentation**:

- **Notebooks** — boilerplate structure, repetitive model-evaluation blocks and notebook reorganisation.
- **The live app** — the Streamlit code in `app/`, the export scripts in `models/` that rebuild each notebook's model and check it against the notebook's stored results, the automated tests, and the deployment configuration.
- **Documentation** — the formatting of this README, `PROGRESS.md`, the folder READMEs and `app/DEPLOYMENT.md`.
- **Fact-checking** — numbers quoted in the app and in this README were verified against the notebooks, the saved models and the datasets. For the app's explanatory notes, Claude assembled verified facts and suggested draft wording, which the team reviewed and edited.

All **dataset analysis, feature-engineering decisions and methodological choices** — including the group-wise battery split, the early-window feature design, the SOH target definition, the CAFV target-proxy exclusion, and the choice and evaluation of every model in all three tracks — are the team's own work, as are the notebooks' written interpretations.

---

<div align="center">

**23CSE301 — Machine Learning · Capstone Project**

B.Tech. Computer Science and Engineering, III Year · Academic Year 2026–27

Amrita Vishwa Vidyapeetham

<sub>⚡ Built with scikit-learn, Pandas, Streamlit and a lot of battery-cycle CSVs · <a href="https://chargeiq.streamlit.app">chargeiq.streamlit.app</a></sub>

</div>
