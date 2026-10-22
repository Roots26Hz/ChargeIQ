# 📦 `models/`

The three trained models used by the [app](../app), one per track.

None of the notebooks saved its trained model, so each export script rebuilds it by **replaying the notebook's own cells verbatim**, then checks the result against the values the notebook stored in its outputs. If any check fails, it writes nothing.

| Track | Script | Model | Checked against the notebook |
|:--|:--|:--|:--|
| 🚗 Classification | `export_classification.py` | Decision Tree, `max_depth=7`, inside its preprocessing pipeline | depth sweep (cell 45); all five test metrics and the confusion matrix `[[47509, 5], [0, 11325]]` (cell 46) |
| 🔋 Regression | `export_regression.py` | Gradient Boosting, with `StandardScaler` and the `cycle_index²` step | row and battery counts; split; best parameters and CV R² at full precision; R² 0.8153 · RMSE 8.0449 · MAE 5.6427; the five GroupKFold folds |
| 🔌 Clustering | `export_clustering.py` | K-Means, K = 3, after imputation, Yeo–Johnson and scaling | all 100,000 stored labels; sizes; Silhouette, Davies–Bouldin, Calinski–Harabasz; profiles; PCA variance |

Each script writes two files:

- `<track>_model.joblib` — a scikit-learn `Pipeline` that loads with no custom code.
- `<track>_metadata.json` — input order and ranges, the notebook's reported results with their cell numbers, provenance, and the library versions it was written with.

## Rebuilding

From the repository root:

```bash
python models/export_classification.py --check   # verify only, write nothing
python models/export_classification.py           # verify, then write
```

The same works for `export_regression.py` (needs the local NASA data in `data/Regression_Dataset/`) and `export_clustering.py` (runs from the committed 100,000-session sample). Each export is deterministic: running it twice writes identical files.

> [!IMPORTANT]
> A saved scikit-learn model is only guaranteed to load under the version that wrote it, **1.9.0**. The app refuses to load a model under any other version, and `app/requirements.txt` pins it.
