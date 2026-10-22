"""Driver track (clustering.ipynb) — the team's own words and reported numbers.

Each quote was extracted from its cell programmatically, so it is verbatim down to the
notebook's own characters; app/tests/test_content.py re-checks every one.
"""

from . import Fact, Quote

NB = "clustering.ipynb"

# ---- Chapter 8: Meet the data

FEATURES = Quote(
    NB, 1,
    "The clustering features are `connectionTime_decimal`, `chargingDuration`, and `kWhDelivered`. "
    "Identifier or index columns are excluded from model fitting. `dayIndicator` is not used because "
    "this analysis focuses on connection time, charging duration, and delivered energy.",
)

SAMPLE = Quote(
    NB, 1,
    "The notebook uses a reproducible sample of up to **100,000 charging records** for model fitting and "
    "evaluation, depending on the number of usable records in the CSV.",
)

SAMPLE_SCOPE = Quote(
    NB, 1,
    "Results describe the selected sample and may differ from results obtained using the full dataset.",
)

OUTLIERS = Quote(
    NB, 6,
    "Potential outliers are not automatically removed because they may represent genuine EV charging "
    "sessions.",
)

PARTIAL_ROWS = Quote(
    NB, 8,
    "Rows with individual missing feature values are retained for median imputation during preprocessing.",
)

EDA = Quote(
    NB, 10,
    "Histograms help reveal skewed distributions and long tails, while the correlation heatmap "
    "summarizes linear relationships between connection time, charging duration, and energy delivered. "
    "These plots support understanding of the data; they do not determine cluster labels or establish "
    "causal relationships.",
)

# ---- Chapter 9: How the model groups

PREPROCESSING = Quote(
    NB, 12,
    "The transformation can reduce the influence of skewed feature distributions, while standardization "
    "places the features on comparable scales for distance-based clustering.",
)

ELBOW = Quote(
    NB, 12,
    "Inertia generally decreases as the number of clusters increases; the elbow curve helps identify a "
    "candidate cluster count where additional clusters provide diminishing improvement.",
)

K_CHOICE = Quote(
    NB, 14,
    "The value of 3 in the code is an initial setting, not a guaranteed optimum. The same number of "
    "clusters is used for hierarchical clustering so that the two algorithms can be compared on a "
    "consistent basis.",
)

PROFILES = Quote(
    NB, 30,
    "This cell summarizes the clusters using the original values of connection time, charging duration, "
    "and energy delivered, rather than the transformed and standardized values used for model fitting.",
)

ARBITRARY_LABELS = Quote(
    NB, 30,
    "Cluster numbers are arbitrary labels and do not imply an inherent ranking.",
)

LINKAGES = Quote(
    NB, 16,
    "These linkage comparisons are for visualization; the main Agglomerative model is fitted separately "
    "in the later cell.",
)

DENDROGRAM = Quote(
    NB, 18,
    "Larger vertical gaps between merges may suggest candidate cut points for forming clusters.",
)

WARD = Quote(
    NB, 20,
    "Ward linkage is used because it seeks to minimize the increase in within-cluster variance at each "
    "merge.",
)

CONNECTIVITY = Quote(
    NB, 20,
    "A sparse nearest-neighbor connectivity graph is used to make the larger experiment more "
    "computationally manageable.",
)

SILHOUETTE = Quote(
    NB, 22,
    "**Silhouette Score:** values closer to 1 generally indicate better-defined clusters; values near 0 "
    "suggest overlapping clusters, and negative values may indicate assignments to inappropriate "
    "clusters.",
)

DAVIES_BOULDIN = Quote(
    NB, 22,
    "**Davies–Bouldin Index:** lower values generally indicate more compact, better-separated clusters.",
)

CALINSKI_HARABASZ = Quote(
    NB, 22,
    "**Calinski–Harabasz Index:** higher values generally indicate greater separation between clusters "
    "relative to within-cluster dispersion.",
)

INTERNAL_METRICS = Quote(
    NB, 22,
    "These are internal evaluation metrics, not ground-truth accuracy scores.",
)

PCA = Quote(
    NB, 24,
    "PCA is used here for visualization only; K-Means is fitted using all three preprocessed features.",
)

PCA_COMPARISON = Quote(
    NB, 26,
    "The projection is intended to help inspect visible grouping and overlap; it does not replace the "
    "numerical evaluation metrics.",
)

TSNE = Quote(
    NB, 28,
    "Distances between widely separated groups, apparent cluster sizes, and global geometry should be "
    "interpreted cautiously. This is a visualization of the existing K-Means assignments, not a separate "
    "clustering model.",
)

PREFERRED = Quote(
    NB, 36,
    "The preferred algorithm should be identified from the actual metric results and visualizations.",
)

NO_GROUND_TRUTH = Quote(
    NB, 36,
    "Since the dataset does not provide ground-truth cluster labels, the discovered groups describe "
    "patterns in the observed charging sessions rather than objectively correct categories.",
)

# ---- Facts

# The only statement in the repository that the data is synthetic is the file the notebook loads.
SOURCE_FILE = Fact(NB, 5, "SYNTHETIC_EV_DATA.csv", "SYNTHETIC_EV_DATA.csv")
ROWS_RETAINED = Fact(NB, 9, "1,965,239", "Rows retained for clustering: 1965239")
DUPLICATES = Fact(NB, 7, "0", "Duplicate rows: 0")

# ---- How the three features are shown. The notebook does not state units; the dataset's
# source does (Gholizadeh, "Electric Vehicle Charging Dataset", Mendeley Data, 2024,
# doi:10.17632/5zrtmp7gwd.1): connection time ranges 0-24, duration is in hours, energy in kWh.

LABELS = {
    "connectionTime_decimal": "Connection time",
    "chargingDuration": "Charging duration",
    "kWhDelivered": "Energy delivered",
}
UNITS = {"connectionTime_decimal": "hour of day", "chargingDuration": "hours", "kWhDelivered": "kWh"}


def clock(hours: float) -> str:
    """17.835 -> '17:50'. The day's last instant, 24.0, shows as '24:00'."""
    minutes = round(hours * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def show(field: str, value: float) -> str:
    if field == "connectionTime_decimal":
        return clock(value)
    if field == "chargingDuration":
        return f"{value:.2f} h"
    return f"{value:.1f} kWh"

QUOTES = (
    FEATURES, SAMPLE, SAMPLE_SCOPE, OUTLIERS,
    PARTIAL_ROWS, EDA, PREPROCESSING, ELBOW,
    K_CHOICE, PROFILES, ARBITRARY_LABELS, LINKAGES,
    DENDROGRAM, WARD, CONNECTIVITY, SILHOUETTE,
    DAVIES_BOULDIN, CALINSKI_HARABASZ, INTERNAL_METRICS, PCA,
    PCA_COMPARISON, TSNE, PREFERRED, NO_GROUND_TRUTH,
)
FACTS = (SOURCE_FILE, ROWS_RETAINED, DUPLICATES)
