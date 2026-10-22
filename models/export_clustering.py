#!/usr/bin/env python3
"""Export the final Clustering model fitted in notebooks/clustering.ipynb.

The notebook never saved its fitted model, so this script reproduces it by
replaying the notebook's own cells verbatim:

    cell 13  median imputer -> Yeo-Johnson -> StandardScaler; K-Means elbow, K = 2..10
    cell 15  KMeans(n_clusters=3, n_init=5, max_iter=100, random_state=42)
    cell 23  Silhouette (5,000-row subset), Davies-Bouldin and Calinski-Harabasz (all rows)
    cell 25  PCA to two components
    cell 31  cluster profiles in original units

The notebook sampled 100,000 sessions from SYNTHETIC_EV_DATA.csv, a 1,965,239-row
file that is not in the repository, and cell 35 saved that exact sample — in the
same order, with both algorithms' labels — as notebooks/EV_clustering_sample_with_labels.csv.
The replay starts from that file, so it runs on any machine.

Nothing is retuned or changed. Before writing anything, the replay is checked
against fingerprints taken from the notebook:

    1. all 100,000 stored K-Means labels, one by one
    2. cell 15  K-Means cluster sizes
    3. cell 23  K-Means Silhouette and Davies-Bouldin to 4 decimals, Calinski-Harabasz (~84,854)
                to a relative 1e-6; all three against EV_clustering_evaluation_metrics.csv to 1e-6
    4. cell 31  K-Means cluster profiles, to 3 decimals
    5. cell 25  variance explained by two PCA components

Agglomerative clustering has no predict(), so it is not exported. Its stored labels
are checked against cell 21's sizes, cell 23's metrics and cell 31's profiles instead.

The notebook's eight figures exist only inside its stored outputs; they are written
byte for byte to assets/figures/clustering/.

If any fingerprint disagrees the script exits non-zero and writes nothing.

Usage (from the repository root):
    python models/export_clustering.py           # verify, then write
    python models/export_clustering.py --check   # verify only
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "clustering.ipynb"
SAMPLE = REPO / "notebooks" / "EV_clustering_sample_with_labels.csv"
METRICS_CSV = REPO / "notebooks" / "EV_clustering_evaluation_metrics.csv"
OUT_MODEL = REPO / "models" / "clustering_model.joblib"
OUT_META = REPO / "models" / "clustering_metadata.json"
FIGURES_DIR = REPO / "assets" / "figures" / "clustering"

RANDOM_STATE = 42
FEATURES = ["connectionTime_decimal", "chargingDuration", "kWhDelivered"]  # cell 7
BEST_K = 3  # cell 15
K_VALUES = range(2, 11)  # cell 13: range(2, min(10, len(X_scaled) - 1) + 1)
METRIC_SAMPLE_SIZE = 5000  # cell 23
MAP_POINTS = 2000  # sessions kept, projected onto the PCA plane, for the app's map
KMEANS = "K-Means"
AGGLOMERATIVE = "Agglomerative Clustering (Ward + kNN connectivity)"

# (cell, index of the image among that cell's image outputs) -> file under assets/figures/clustering/
FIGURES = {
    "eda/01_feature_distributions.png": (11, 0),
    "eda/02_correlation_heatmap.png": (11, 1),
    "models/01_kmeans_elbow_curve.png": (13, 0),
    "models/02_linkage_comparison_dendrograms.png": (17, 0),
    "models/03_ward_dendrogram.png": (19, 0),
    "models/04_kmeans_pca.png": (25, 0),
    "models/05_agglomerative_pca.png": (27, 0),
    "models/06_kmeans_tsne.png": (29, 0),
}


class FingerprintMismatch(RuntimeError):
    """The replay does not reproduce the notebook's stored results."""


# ---------------------------------------------------------------- notebook outputs


def _notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _find_cell(cells: list[dict], marker: str) -> tuple[int, str]:
    """Locate a code cell by a phrase in its source; return (index, its text output)."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            parts = []
            for out in cell.get("outputs", []):
                text = out.get("text") or (out.get("data") or {}).get("text/plain", "")
                parts.append("".join(text))
            return i, "\n".join(parts)
    raise FingerprintMismatch(f"no notebook cell contains {marker!r} — has the notebook changed?")


def _sizes(text: str) -> dict[int, int]:
    return {int(label): int(n) for label, n in re.findall(r"^(\d+)\s+(\d+)\s*$", text, re.M)}


def _profiles(text: str) -> dict[int, list[float]]:
    rows = re.findall(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", text, re.M)
    return {int(label): [float(a), float(b), float(c)] for label, a, b, c in rows}


def read_notebook_results() -> dict:
    """Parse the values this script must reproduce, plus the notebook's reported results."""
    cells = _notebook_cells()

    i5, out5 = _find_cell(cells, "SYNTHETIC_EV_DATA.csv")
    rows, columns = map(int, re.search(r"Dataset shape:\s*\((\d+),\s*(\d+)\)", out5).groups())
    source_file = re.search(r'CSV_PATH = r?"[^"]*?([^"\\/]+\.csv)"', "".join(cells[i5]["source"])).group(1)

    i7, out7 = _find_cell(cells, "Descriptive statistics for clustering features")
    stats = {}
    for feature in FEATURES:
        first = re.search(rf"^{feature}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", out7, re.M)
        second = re.search(rf"^{feature}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", out7, re.M)
        if not first or not second:
            raise FingerprintMismatch(f"could not read the statistics for {feature} (cell {i7})")
        keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        stats[feature] = dict(zip(keys, map(float, first.groups() + second.groups())))

    i13, out13 = _find_cell(cells, "SAMPLE_SIZE = 100000")
    sample_rows = int(re.search(r"Rows used for clustering:\s*(\d+)", out13).group(1))

    i15, out15 = _find_cell(cells, "kmeans_labels = kmeans_model.fit_predict(X_scaled)")
    i21, out21 = _find_cell(cells, "agg_labels = agg_model.fit_predict(X_scaled)")
    neighbours = int(re.search(r"k-nearest neighbors =\s*(\d+)", out21).group(1))

    i23, out23 = _find_cell(cells, "def evaluate_clustering(")
    metrics = {}
    for index, name in ((0, KMEANS), (1, AGGLOMERATIVE)):
        numbers = [float(n) for line in re.findall(rf"^{index}\s+(.*)$", out23, re.M)
                   for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", line)]
        if len(numbers) != 4:
            raise FingerprintMismatch(f"could not read the metrics for {name} (cell {i23})")
        metrics[name] = {"rows": int(numbers[0]), "silhouette": numbers[1], "davies_bouldin": numbers[2],
                         "calinski_harabasz": numbers[3]}

    i25, out25 = _find_cell(cells, "pca = PCA(n_components=2")
    pca_variance = float(re.search(r"PCA components:\s*([\d.]+)%", out25).group(1))

    i31, out31 = _find_cell(cells, 'profiles.groupby("KMeans_Cluster")')
    kmeans_part, agg_part = out31.split("Hierarchical cluster profiles")

    full = pd.read_csv(METRICS_CSV)
    full_precision = {
        row["Algorithm"]: {"silhouette": float(row[f"Silhouette Score (sample n={METRIC_SAMPLE_SIZE})"]),
                           "davies_bouldin": float(row["Davies-Bouldin Index (all rows)"]),
                           "calinski_harabasz": float(row["Calinski-Harabasz Index (all rows)"])}
        for _, row in full.iterrows()
    }

    results = {
        "source": {"cell": i5, "file": source_file, "rows": rows, "columns": columns},
        "full_dataset_statistics": {"cell": i7, "features": stats},
        "sample": {"cell": i13, "rows": sample_rows},
        "kmeans_sizes": {"cell": i15, "sizes": _sizes(out15)},
        "agglomerative_sizes": {"cell": i21, "sizes": _sizes(out21), "connectivity_neighbours": neighbours},
        "metrics": {"cell": i23, "csv": str(METRICS_CSV.relative_to(REPO)), "rounded": metrics,
                    "full_precision": full_precision},
        "pca": {"cell": i25, "variance_explained_pct": pca_variance},
        "profiles": {"cell": i31, KMEANS: _profiles(kmeans_part), AGGLOMERATIVE: _profiles(agg_part)},
    }
    if (len(results["kmeans_sizes"]["sizes"]) != BEST_K or len(results["agglomerative_sizes"]["sizes"]) != BEST_K
            or any(len(results["profiles"][a]) != BEST_K for a in (KMEANS, AGGLOMERATIVE))
            or set(full_precision) != {KMEANS, AGGLOMERATIVE}):
        raise FingerprintMismatch("notebook outputs did not parse into the expected shapes")
    return results


def notebook_figures() -> dict[str, bytes]:
    """The eight figures, decoded from the notebook's stored image outputs."""
    cells = _notebook_cells()
    figures = {}
    for name, (cell, position) in FIGURES.items():
        images = [o["data"]["image/png"] for o in cells[cell].get("outputs", []) if "image/png" in (o.get("data") or {})]
        figures[name] = base64.b64decode("".join(images[position]))
    return figures


# ---------------------------------------------------------------- replay


def _evaluate(X: np.ndarray, labels: np.ndarray, metric_indices: np.ndarray) -> dict:
    """Cell 23 — verbatim metric definitions."""
    return {
        "rows": int(len(labels)),
        "silhouette": float(silhouette_score(X[metric_indices], labels[metric_indices])),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def _profile(X_original: pd.DataFrame, labels: np.ndarray) -> dict[int, list[float]]:
    means = X_original.assign(label=labels).groupby("label")[FEATURES].mean()
    return {int(label): [float(v) for v in row] for label, row in means.iterrows()}


def replay() -> dict:
    # One thread: multithreaded K-Means sums in whatever order the threads finish, which can move
    # a cluster centre by one unit in the last place between runs. Single-threaded, the export is
    # byte-for-byte reproducible. (The labels and metrics are the same either way.)
    # NumPy 2 on macOS's Accelerate BLAS also raises spurious divide/overflow/invalid warnings inside
    # matmul during K-Means; the results are unaffected, and the fingerprints check them anyway.
    with threadpool_limits(limits=1), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _replay()


def _replay() -> dict:
    sample = pd.read_csv(SAMPLE)
    X_original = sample[FEATURES].copy()

    preprocessor = Pipeline([  # cell 13
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler", StandardScaler()),
    ])
    X_scaled = preprocessor.fit_transform(X_original)
    inertias = {k: float(KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=100)
                         .fit(X_scaled).inertia_) for k in K_VALUES}

    kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=5, max_iter=100)  # cell 15
    kmeans_labels = kmeans.fit_predict(X_scaled)
    agg_labels = sample["Hierarchical_Cluster"].to_numpy()  # cell 21's labels, as saved by cell 35

    rng = np.random.default_rng(RANDOM_STATE)  # cell 23
    metric_indices = np.sort(rng.choice(len(X_scaled), size=METRIC_SAMPLE_SIZE, replace=False))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)  # cell 25
    X_pca = pca.fit_transform(X_scaled)

    pipeline = Pipeline([*preprocessor.steps, ("kmeans", kmeans)])
    return {
        "sample": sample, "X_original": X_original, "X_scaled": X_scaled, "preprocessor": preprocessor,
        "kmeans": kmeans, "pipeline": pipeline, "inertias": inertias, "pca": pca, "X_pca": X_pca,
        "kmeans_labels": kmeans_labels, "agg_labels": agg_labels,
        "labels_match": bool(np.array_equal(kmeans_labels, sample["KMeans_Cluster"].to_numpy())),
        "pipeline_matches": bool(np.array_equal(pipeline.predict(X_original), kmeans_labels)),
        "metrics": {KMEANS: _evaluate(X_scaled, kmeans_labels, metric_indices),
                    AGGLOMERATIVE: _evaluate(X_scaled, agg_labels, metric_indices)},
        "profiles": {KMEANS: _profile(X_original, kmeans_labels), AGGLOMERATIVE: _profile(X_original, agg_labels)},
        "sizes": {KMEANS: {int(k): int(v) for k, v in pd.Series(kmeans_labels).value_counts().items()},
                  AGGLOMERATIVE: {int(k): int(v) for k, v in pd.Series(agg_labels).value_counts().items()}},
    }


def check_fingerprints(result: dict, notebook: dict) -> list[str]:
    problems = []

    def compare(what, got, expected):
        if got != expected:
            problems.append(f"{what}: replay {got} vs notebook {expected}")

    compare("sample rows", len(result["X_original"]), notebook["sample"]["rows"])
    if not result["labels_match"]:
        mismatched = int((result["kmeans_labels"] != result["sample"]["KMeans_Cluster"].to_numpy()).sum())
        problems.append(f"K-Means labels: {mismatched} of {len(result['kmeans_labels'])} differ from the stored labels")
    if not result["pipeline_matches"]:
        problems.append("the exported pipeline does not reproduce the replayed K-Means labels")

    compare("K-Means sizes", result["sizes"][KMEANS], notebook["kmeans_sizes"]["sizes"])
    compare("Agglomerative sizes (stored labels)", result["sizes"][AGGLOMERATIVE],
            notebook["agglomerative_sizes"]["sizes"])

    for algorithm in (KMEANS, AGGLOMERATIVE):
        got, rounded = result["metrics"][algorithm], notebook["metrics"]["rounded"][algorithm]
        full = notebook["metrics"]["full_precision"][algorithm]
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            if key == "calinski_harabasz":
                # ~84,854: four decimals are nine significant figures, beyond what floating-point
                # summation reproduces across machines (the stored Agglomerative labels drift the
                # same way), so it is compared to a relative 1e-6 like the full-precision values.
                if not math.isclose(got[key], rounded[key], rel_tol=1e-6):
                    problems.append(f"{algorithm} {key}: replay {got[key]:.4f} vs notebook {rounded[key]}")
            else:
                compare(f"{algorithm} {key}", round(got[key], 4), rounded[key])
            if not math.isclose(got[key], full[key], rel_tol=1e-6):
                problems.append(f"{algorithm} {key}: replay {got[key]!r} vs {METRICS_CSV.name} {full[key]!r}")
        compare(f"{algorithm} profiles", {k: [round(v, 3) for v in p] for k, p in result["profiles"][algorithm].items()},
                notebook["profiles"][algorithm])

    compare("PCA variance explained (%)", round(result["pca"].explained_variance_ratio_.sum() * 100, 2),
            notebook["pca"]["variance_explained_pct"])
    return problems


# ---------------------------------------------------------------- metadata


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(result: dict, notebook: dict) -> dict:
    X_original, kmeans, preprocessor, pca = result["X_original"], result["kmeans"], result["preprocessor"], result["pca"]

    centres = pd.DataFrame(preprocessor.named_steps["power"].inverse_transform(
        preprocessor.named_steps["scaler"].inverse_transform(kmeans.cluster_centers_)), columns=FEATURES)

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = np.sort(rng.choice(len(X_original), size=MAP_POINTS, replace=False))
    map_points = [
        {"pc1": round(float(result["X_pca"][i, 0]), 4), "pc2": round(float(result["X_pca"][i, 1]), 4),
         "cluster": int(result["kmeans_labels"][i])}
        for i in chosen
    ]

    def per_algorithm(algorithm: str) -> dict:
        return {
            "sizes": {str(k): v for k, v in sorted(result["sizes"][algorithm].items())},
            "profiles_mean": {str(k): dict(zip(FEATURES, v)) for k, v in sorted(result["profiles"][algorithm].items())},
            "metrics": result["metrics"][algorithm],
        }

    return {
        "track": "clustering",
        "provenance": {
            "notebook": "notebooks/clustering.ipynb",
            "notebook_sha256": _sha256(NOTEBOOK),
            "sample": str(SAMPLE.relative_to(REPO)),
            "sample_sha256": _sha256(SAMPLE),
            "source_file": notebook["source"]["file"],
            "source_rows": notebook["source"]["rows"],
            "method": (
                "The notebook did not save its fitted model. This artifact was produced by "
                "replaying the notebook's preprocessing and K-Means cells verbatim on the exact "
                "100,000-session sample the notebook saved, and was written only after the replay "
                "reproduced every stored K-Means label and the notebook's stored results exactly."
            ),
            "replayed_cells": [13, 15, 23, 25, 31],
            "fingerprints_verified": [
                "all 100,000 stored K-Means labels",
                f"cell {notebook['kmeans_sizes']['cell']}: K-Means cluster sizes",
                f"cell {notebook['metrics']['cell']}: Silhouette, Davies-Bouldin, Calinski-Harabasz, "
                f"and {METRICS_CSV.name} to a relative 1e-6",
                f"cell {notebook['profiles']['cell']}: cluster profiles, 3 decimal places",
                f"cell {notebook['pca']['cell']}: PCA variance explained",
                "Agglomerative: stored labels reproduce the notebook's sizes, metrics and profiles",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "feature_order": FEATURES,
        "sample_ranges": {
            f: {"min": float(X_original[f].min()), "max": float(X_original[f].max()),
                **{f"p{q}": float(X_original[f].quantile(q / 100)) for q in (1, 25, 50, 75, 99)}}
            for f in FEATURES
        },
        "dataset": {"source_rows": notebook["source"]["rows"], "sample_rows": int(len(X_original)),
                    "excluded_columns": ["dayIndicator"]},
        "kmeans": {
            "params": {"n_clusters": BEST_K, "n_init": 5, "max_iter": 100, "random_state": RANDOM_STATE},
            **per_algorithm(KMEANS),
            "centres_original_units": {str(k): dict(zip(FEATURES, map(float, row)))
                                       for k, row in centres.iterrows()},
            "elbow_inertia": {str(k): v for k, v in result["inertias"].items()},
        },
        "agglomerative": {
            "params": {"n_clusters": BEST_K, "linkage": "ward",
                       "connectivity_neighbours": notebook["agglomerative_sizes"]["connectivity_neighbours"]},
            **per_algorithm(AGGLOMERATIVE),
        },
        "pca": {
            "components": pca.components_.tolist(),
            "mean": pca.mean_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "map_points": map_points,
        },
        "figures": {name: {"cell": cell} for name, (cell, _) in FIGURES.items()},
        "notebook_results": notebook,
    }


# ---------------------------------------------------------------- writing


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, so a failure never leaves a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    umask = os.umask(0)
    os.umask(umask)
    try:
        write(Path(tmp))
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates 0600; give the artifact normal permissions
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify fingerprints, write nothing")
    args = parser.parse_args(argv)

    print(f"Reading expected results from {NOTEBOOK.relative_to(REPO)} ...")
    notebook = read_notebook_results()
    print(f"Replaying preprocessing and K-Means on {SAMPLE.relative_to(REPO)} ...")
    result = replay()

    problems = check_fingerprints(result, notebook)
    if problems:
        print("\nFINGERPRINT MISMATCH — nothing was written:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    m = result["metrics"][KMEANS]
    print("\nAll fingerprints match the notebook:")
    print(f"  labels         all {len(result['kmeans_labels']):,} stored K-Means labels reproduced")
    print(f"  sizes          {dict(sorted(result['sizes'][KMEANS].items()))}")
    print(f"  metrics        silhouette {m['silhouette']:.4f}  DB {m['davies_bouldin']:.4f}  "
          f"CH {m['calinski_harabasz']:.4f}")
    print(f"  PCA            {result['pca'].explained_variance_ratio_.sum() * 100:.2f}% variance in two components")
    print("  Agglomerative  stored labels reproduce the notebook's sizes, metrics and profiles")

    if args.check:
        print("\n--check: verification only, nothing written.")
        return 0

    metadata = build_metadata(result, notebook)
    _atomic_write(OUT_MODEL, lambda p: joblib.dump(result["pipeline"], p, compress=3))
    _atomic_write(OUT_META, lambda p: p.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))
    for name, png in notebook_figures().items():
        _atomic_write(FIGURES_DIR / name, lambda p, png=png: p.write_bytes(png))

    print(f"\nWrote {OUT_MODEL.relative_to(REPO)}  ({OUT_MODEL.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {OUT_META.relative_to(REPO)}  ({OUT_META.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {len(FIGURES)} figures to {FIGURES_DIR.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
