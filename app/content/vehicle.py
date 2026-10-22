"""Vehicle track (classification.ipynb) — the team's own words and reported numbers."""

from . import Example, Fact, Quote

NB = "classification.ipynb"

# ---- Chapter 2: Meet the data

CLASS_IMBALANCE = Quote(
    NB, 10,
    "The target distribution shows that Battery Electric Vehicles (BEVs) form the majority class, while "
    "Plug-in Hybrid Electric Vehicles (PHEVs) form the minority class. Therefore, the target variable is "
    "moderately imbalanced. Model performance should not be judged using accuracy alone; weighted "
    "F1-score and the confusion matrix will also be considered during model evaluation.",
)

FEATURE_TARGET_OBSERVATIONS = Quote(
    NB, 16,
    "Electric Range shows a noticeable difference between the two vehicle types, suggesting that it may be "
    "an informative feature for classification. Model Year also shows differences in the distribution of "
    "the two classes, indicating that vehicle age may provide useful information for predicting Electric "
    "Vehicle Type. These observations are exploratory and will be verified quantitatively through the "
    "classification models.",
)

CLEANING = Quote(
    NB, 20,
    "The missing values were handled using median imputation for numerical features and mode imputation "
    "for categorical features. This preserves the available observations while providing a simple and "
    "robust treatment for the small number of missing values.",
)

OUTLIERS = Quote(
    NB, 20,
    "Potential numerical outliers were identified using the IQR method. These values were retained because "
    "extreme vehicle characteristics may represent valid observations rather than data-entry errors.",
)

IDENTIFIERS = Quote(
    NB, 21,
    "Identifier columns such as VIN (1-10) and DOL Vehicle ID are excluded because they identify individual "
    "records rather than describe meaningful vehicle characteristics.",
)

VEHICLE_AGE = Quote(
    NB, 23,
    "Vehicle age may provide additional information about the type of electric vehicle because BEVs and "
    "PHEVs may have different adoption patterns across model years.",
)

SPLIT = Quote(
    NB, 28,
    "The dataset was divided into training and testing sets using an 80:20 split. Stratification preserves "
    "approximately the same BEV-to-PHEV class proportion in both sets.",
)

STRATIFIED_SPLIT = Quote(
    NB, 25,
    "Stratification is used to preserve the proportion of BEV and PHEV classes in both the training and "
    "testing sets.",
)

# ---- Chapter 3: How the model thinks

TRAIN_ONLY_FITTING = Quote(
    NB, 29,
    "The preprocessing transformations are fitted only on the training data.The same fitted transformations "
    "are then applied to the test data. This prevents information from the test set from leaking into the "
    "training process.",
)

TREE_TUNING = Quote(
    NB, 44,
    "The max_depth parameter is tuned using a stratified validation subset created only from the 60,000 "
    "training samples. The best depth is selected based on weighted F1 score.",
)

FINAL_TREE = Quote(
    NB, 44,
    "After selecting the best depth, the final Decision Tree is trained using all 60,000 training samples "
    "and evaluated on the unchanged test set.",
)

FAIR_COMPARISON = Quote(
    NB, 56,
    "The same training and test split and the preprocessed predictor features are reused so that the "
    "models can be compared fairly.",
)

COMPARISON_CAUTION = Quote(
    NB, 82,
    "Near-perfect results across several different models are a reason to audit possible target-related "
    "features, duplicates, and train/test separation, not a reason to assume all models will be perfect "
    "in deployment.",
)

CROSS_VALIDATION = Quote(
    NB, 85,
    "The means are very high and variability is small on this subset, but cross-validation does not "
    "replace a separate external test or a leakage audit.",
)

RANGE_AND_MODEL_CAUTION = Quote(
    NB, 64,
    "Because vehicle range and model information are closely related to vehicle type, verify that they "
    "are appropriate inputs for the intended use case.",
)

SELECTION = Quote(
    NB, 87,
    "The saved comparison identifies Decision Tree as the best model by weighted F1 and accuracy, while "
    "AdaBoost has the highest ROC-AUC after rounding. Decision Tree and Bagging both have weighted F1 "
    "rounded to 0.9999, so their practical difference is extremely small in this evaluation.",
)

LEAKAGE_CAUTION = Quote(
    NB, 87,
    "Before making a strong real-world claim, review the confusion matrices, verify that the target was not "
    "indirectly used to construct predictors, check duplicates and train/test overlap, and consider whether "
    "features such as Electric Range, Make, and Model would be available at the time the prediction is "
    "meant to be made. Very high test scores can be genuine, but they warrant a careful leakage and split "
    "audit.",
)

# ---- Numbers the exported metadata does not carry

TRAINING_POOL_ROWS = Fact(NB, 26, "235,354", "Training set shape: (235354, 13)")
MODEL_TRAINING_ROWS = Fact(NB, 30, "60,000", "Processed training shape: (60000, 1694)")
TARGET_PROXY_DROP = Fact(NB, 22, "Clean Alternative Fuel Vehicle (CAFV) Eligibility", "target_proxy_columns")

QUOTES = (
    CLASS_IMBALANCE, FEATURE_TARGET_OBSERVATIONS, CLEANING, OUTLIERS, IDENTIFIERS, VEHICLE_AGE, SPLIT,
    STRATIFIED_SPLIT, TRAIN_ONLY_FITTING, TREE_TUNING, FINAL_TREE, FAIR_COMPARISON, COMPARISON_CAUTION, CROSS_VALIDATION,
    RANGE_AND_MODEL_CAUTION, SELECTION, LEAKAGE_CAUTION,
)
FACTS = (TRAINING_POOL_ROWS, MODEL_TRAINING_ROWS, TARGET_PROXY_DROP)

# ---- Chapter 4: ready-made inputs (verified against the replayed split by app/tests/test_content.py)

EXAMPLES = (
    Example("model_y", "Tesla Model Y 2023", "TESLA", "MODEL Y", 2023, 0,
            "training", 3777, "BEV", "BEV"),
    Example("model_3", "Tesla Model 3 2018", "TESLA", "MODEL 3", 2018, 215,
            "training", 1146, "BEV", "BEV"),
    Example("volt", "Chevrolet Volt 2017", "CHEVROLET", "VOLT", 2017, 53,
            "training", 240, "PHEV", "PHEV"),
    Example("i3_97", "BMW i3 2017 · range 97", "BMW", "I3", 2017, 97,
            "training", 56, "PHEV", "PHEV"),
    Example("i3_81", "BMW i3 2017 · range 81", "BMW", "I3", 2017, 81,
            "training", 21, "BEV", "BEV"),
    Example("ranger", "Ford Ranger 2000", "FORD", "RANGER", 2000, 58,
            "test", 2, "BEV", "PHEV"),
)
