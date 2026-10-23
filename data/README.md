# 📦 `data/`

Raw datasets for the three ChargeIQ tracks. **Each track has its own dataset — nothing is shared between them.**

> [!IMPORTANT]
> The dataset files are **not committed**. All three subfolders are gitignored (`.gitignore` → `#Datasets`), because they total roughly 770 MB — and the clustering file alone is over GitHub's 100 MB limit. Download them before running any notebook. The live app needs none of them.

**Download:** the regression and classification datasets are available at this [OneDrive link](https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_sc_u4cse24664_cb_students_amrita_edu/IgAhvNV5RuhjTrFM-UakeblJAfpsW5pjQ2QUCYFpDwVBYF4?e=5itzGU).

## Expected layout

```
data/
├── README.md                                  ← this file (the only tracked item here)
├── Regression_Dataset/                        ← 🔋 regression track only   (~575 MB)
│   ├── metadata.csv                              7,565 × 10 — index of every cycle
│   ├── data/                                     7,565 per-cycle curve CSVs
│   └── extra_infos/                              9 NASA README files, one per battery group
├── Classification_Dataset/                    ← 🚗 classification track only  (~77 MB)
│   └── Electric_Vehicle_Population_Data.csv      294,193 × 16
└── Clustering_Dataset/                        ← 🔌 clustering track only      (~115 MB)
    └── SYNTHETIC_EV_DATA.csv                     1,965,239 × 4
```

Recreate this structure exactly — the notebooks use these relative paths.

---

## 🔋 `Regression_Dataset/` — NASA Li-ion Battery Aging (PCoE)

Used by [`notebooks/regression.ipynb`](../notebooks/regression.ipynb). **Target: State of Health (`SOH_pct`, %).**

Charge–discharge–impedance cycle data for 34 Li-ion cells run to failure under controlled conditions at NASA Ames Prognostics Center of Excellence. Source: the cleaned Kaggle mirror `patrickfleith/nasa-battery-dataset`.

### `metadata.csv` — 7,565 rows × 10 columns

One row per operation, indexing the per-cycle files through its `filename` column.

| Column | Notes |
|:--|:--|
| `type` | `charge` (2,815) · `discharge` (2,794) · `impedance` (1,956) |
| `start_time` | Timestamp as a 6-element array |
| `ambient_temperature` | Test-protocol temperature group (°C) |
| `battery_id` | Cell identifier, e.g. `B0047` — **the grouping key for the train/test split** |
| `test_id`, `uid` | Run identifiers |
| `filename` | Points into `data/`, e.g. `00001.csv` |
| `Capacity` | Ah — **populated on `discharge` rows only** |
| `Re`, `Rct` | Electrolyte and charge-transfer resistance — **`impedance` rows only** |

The missingness is structural, not corruption: each measurement type only produces its own columns. The notebook resolves this with a structural join in §1b.

### `data/` — 7,565 per-cycle CSVs

The raw measurement curve for one operation each, named by `metadata.csv`'s `filename`. Columns: `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_load`, `Voltage_load`, `Time` (charge rows use `Current_charge` / `Voltage_charge`).

The notebook derives five physics-informed features from these curves, using only the **first 300 seconds** of each discharge — a feature built from the full curve would restate the target, since capacity *is* discharge current integrated over time.

### `extra_infos/` — 9 NASA documentation files

`README_<battery numbers>.txt`, one per battery group, describing each group's charge/discharge/impedance protocol, cut-off voltages and end-of-life criterion. **These are NASA's original files — leave them as shipped.**

> Cells **B0049–B0052** are dropped in §4: 25 cycles each against 28–197 for every other cell, and B0052 is missing 84 % of its capacity readings.

---

## 🚗 `Classification_Dataset/` — Electric Vehicle Population Data (Washington State)

Used by [`notebooks/classification.ipynb`](../notebooks/classification.ipynb). **Target: `Electric Vehicle Type`.**

Official registry of battery-electric and plug-in-hybrid vehicles registered in Washington State. Source: [data.wa.gov](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2/about_data).

### `Electric_Vehicle_Population_Data.csv` — 294,193 rows × 16 columns

| Group | Columns |
|:--|:--|
| Identifiers | `VIN (1-10)`, `DOL Vehicle ID` |
| Location | `County`, `City`, `State`, `Postal Code`, `Legislative District`, `Vehicle Location`, `2020 GEOID`, `Electric Utility` |
| Vehicle | `Model Year`, `Make`, `Model`, `Electric Range` |
| **Target** | **`Electric Vehicle Type`** — BEV **80.75 %** / PHEV **19.25 %** |
| Excluded | `Clean Alternative Fuel Vehicle (CAFV) Eligibility` |

Three columns are dropped before modelling. The two identifiers label a record rather than describe a vehicle. **CAFV eligibility is dropped as a target proxy** — it is derived from vehicle type and electric range, so keeping it would leak the answer.

The class imbalance is why the notebook reports **weighted** precision, recall and F1 alongside accuracy.

---

## 🔌 `Clustering_Dataset/` — Electric Vehicle Charging Dataset *(synthetic)*

Used by [`notebooks/clustering.ipynb`](../notebooks/clustering.ipynb). **Unsupervised — no target.**

> [!IMPORTANT]
> **This data is synthetic.** It was generated with conditional tabular GANs (CTGAN) and kernel density estimation from real charging sessions at a Caltech EV parking lot, extending 185 real days to 29,600 synthetic ones.

**Source:** Gholizadeh, N. (2024). *Electric Vehicle Charging Dataset*. Mendeley Data, V1. [doi:10.17632/5zrtmp7gwd.1](https://doi.org/10.17632/5zrtmp7gwd.1). Licensed **CC BY 4.0** — credit the author when reusing it. Downloaded from its [Kaggle mirror](https://www.kaggle.com/datasets/mexwell/electric-vehicle-charging-dataset), where the file is 120,370,559 bytes.

### `SYNTHETIC_EV_DATA.csv` — 1,965,239 rows × 4 columns

| Column | Meaning | Unit |
|:--|:--|:--|
| `connectionTime_decimal` | When the car was plugged in | hour of day, 0–24 |
| `chargingDuration` | How long it charged | hours |
| `kWhDelivered` | Energy delivered | kWh |
| `dayIndicator` | Day number, 1–29,600 — **not used for clustering** | — |

No missing values and no duplicate rows. The notebook clusters a reproducible **100,000-row sample** (`random_state=42`), saved with both algorithms' labels as [`notebooks/EV_clustering_sample_with_labels.csv`](../notebooks/EV_clustering_sample_with_labels.csv) — which is all the app and its tests need.
