"""
Main approach:
1. Load and validate the dataset.
2. Perform EDA and save useful summaries/plots.
3. Clean the target and remove duplicate/customer-ID columns.
4. Create deterministic business features BEFORE the split.
5. Split data into train/test using stratification.
6. Learn missing-value rules, outlier caps, scaling and encoding ONLY from
   the training set through a single sklearn Pipeline.
7. Train Logistic Regression.
8. Evaluate using Accuracy, Precision, Recall, F1 and ROC-AUC.
9. Save the trained pipeline and all important outputs.

Expected input file:
    customer_churn.csv (also supports customer_churn.csv)

Expected target column:
    Churn
"""
from __future__ import annotations
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
# Accept the filename used by the project. Prefer customer_churn.csv,
# while keeping compatibility with the original customer_churnn.csv name.
_CANDIDATE_DATA_FILES = [
    BASE_DIR / "customer_churn.csv",
    BASE_DIR / "customer_churnn.csv",
]
DATA_FILE = next((path for path in _CANDIDATE_DATA_FILES if path.exists()), _CANDIDATE_DATA_FILES[0])
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

def section(title: str) -> None:
    """Print a consistent section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_dataframe(data: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame in the output directory."""
    path = OUTPUT_DIR / filename
    data.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_plot(filename: str) -> None:
    """Save and close the current matplotlib figure."""
    path = VISUALIZATION_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Cap numerical values using IQR boundaries learned from training data.

    This transformer is intentionally position-based because it is placed
    AFTER SimpleImputer. SimpleImputer returns a NumPy array, not a
    DataFrame, so the transformer must not depend on X.columns.
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("IQRCapper expects a 2-dimensional numeric array.")

        self.n_features_in_ = X.shape[1]

        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.multiplier * iqr
        self.upper_bounds_ = q3 + self.multiplier * iqr

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()

        if X.ndim != 2:
            raise ValueError("IQRCapper expects a 2-dimensional numeric array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "IQRCapper received a different number of features than during fit."
            )

        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

def load_dataset() -> pd.DataFrame:
    section("1. LOADING DATASET")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected one of:\n"
            f"  - {BASE_DIR / 'customer_churn.csv'}\n"
            f"  - {BASE_DIR / 'customer_churnn.csv'}"
        )

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if "Churn" not in df.columns:
        raise ValueError("Required target column 'Churn' was not found.")

    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df

def run_eda(df: pd.DataFrame) -> None:
    section("2. EXPLORATORY DATA ANALYSIS")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset information:")
    df.info()

    # Missing values
    missing = (
        df.isnull()
        .sum()
        .rename("Missing_Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    save_dataframe(missing, "missing_values.csv")

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_summary = pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Duplicate Rows",
                "Unique Rows",
            ],
            "Value": [
                len(df),
                duplicate_count,
                len(df.drop_duplicates()),
            ],
        }
    )
    save_dataframe(duplicate_summary, "duplicate_analysis.csv")

    # Unique values
    unique_values = (
        df.nunique(dropna=False)
        .rename("Unique_Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    save_dataframe(unique_values, "unique_values.csv")

    # Descriptive statistics
    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    if numeric_columns:
        statistics = df[numeric_columns].describe().T.reset_index()
        statistics = statistics.rename(columns={"index": "Feature"})
        save_dataframe(statistics, "numerical_statistics.csv")

    # Categorical distributions
    categorical_columns = [
        column
        for column in df.columns
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_categorical_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]

    categorical_rows = []
    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False)
        for category, count in counts.items():
            categorical_rows.append(
                {
                    "Column": column,
                    "Category": category,
                    "Count": int(count),
                }
            )

    if categorical_rows:
        save_dataframe(
            pd.DataFrame(categorical_rows),
            "categorical_summary.csv",
        )

    # Target distribution
    churn = clean_target(df["Churn"])
    churn_summary = (
        churn.value_counts()
        .sort_index()
        .rename_axis("Churn")
        .reset_index(name="Customer_Count")
    )
    churn_summary["Percentage"] = (
        churn_summary["Customer_Count"] / len(churn) * 100
    )
    save_dataframe(churn_summary, "churn_distribution.csv")

    churn_rate = churn.mean() * 100
    print(f"\nOverall churn rate: {churn_rate:.2f}%")

    # Useful numerical plots
    for column, filename, title, xlabel in [
        (
            "Tenure",
            "tenure_distribution.png",
            "Customer Tenure Distribution",
            "Tenure (Months)",
        ),
        (
            "MonthlyCharges",
            "monthly_charges_distribution.png",
            "Monthly Charges Distribution",
            "Monthly Charges",
        ),
        (
            "TotalCharges",
            "total_charges_distribution.png",
            "Total Charges Distribution",
            "Total Charges",
        ),
    ]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                plt.figure(figsize=(8, 5))
                plt.hist(values, bins=20)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel("Number of Customers")
                save_plot(filename)

    # Contract distribution
    if "Contract" in df.columns:
        counts = df["Contract"].value_counts()
        plt.figure(figsize=(8, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title("Contract Type Distribution")
        plt.xlabel("Contract Type")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=20, ha="right")
        save_plot("contract_distribution.png")

def clean_target(series: pd.Series) -> pd.Series:
    """Convert common churn labels to 0/1."""
    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    return cleaned

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    section("3. DATA CLEANING AND FEATURE ENGINEERING")

    data = df.copy()

    # Remove exact duplicate rows.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(data)}")

    # CustomerID is an identifier, not a predictive feature.
    if "CustomerID" in data.columns:
        data = data.drop(columns=["CustomerID"])
        print("CustomerID removed.")

    # Convert target to binary.
    data["Churn"] = clean_target(data["Churn"])

    invalid_target_rows = int(data["Churn"].isna().sum())
    if invalid_target_rows:
        print(f"Rows with invalid Churn labels removed: {invalid_target_rows}")
        data = data.dropna(subset=["Churn"])

    data["Churn"] = data["Churn"].astype(int)

    # Convert known numeric columns safely.
    for column in ["Tenure", "MonthlyCharges", "TotalCharges"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # target leakage.

    if {"MonthlyCharges", "Tenure"}.issubset(data.columns):
        data["CustomerLifetimeValue"] = (
            data["MonthlyCharges"] * data["Tenure"]
        )

    if {"TotalCharges", "Tenure", "MonthlyCharges"}.issubset(data.columns):
        data["AverageMonthlySpend"] = np.where(
            data["Tenure"] > 0,
            data["TotalCharges"] / data["Tenure"],
            data["MonthlyCharges"],
        )

    if "Tenure" in data.columns:
        data["TenureGroup"] = pd.cut(
            data["Tenure"],
            bins=[-np.inf, 6, 12, 24, 48, np.inf],
            labels=[
                "New Customer",
                "Short Term",
                "Medium Term",
                "Long Term",
                "Loyal Customer",
            ],
        )

    if "MonthlyCharges" in data.columns:
        data["MonthlyChargeCategory"] = pd.cut(
            data["MonthlyCharges"],
            bins=[-np.inf, 35, 70, np.inf],
            labels=[
                "Low Charge",
                "Medium Charge",
                "High Charge",
            ],
        )

    # Count actual subscribed services.
    service_columns = [
        column
        for column in [
            "PhoneService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if column in data.columns
    ]

    if service_columns:
        data["ServiceEngagement"] = (
            data[service_columns]
            .apply(
                lambda row: sum(
                    str(value).strip().lower() == "yes"
                    for value in row
                ),
                axis=1,
            )
            .astype(int)
        )

    if "SeniorCitizen" in data.columns:
        data["SeniorCustomer"] = (
            pd.to_numeric(
                data["SeniorCitizen"],
                errors="coerce",
            )
            .map({1: "Senior", 0: "Non-Senior"})
        )

    engineered_features = [
        column
        for column in [
            "CustomerLifetimeValue",
            "AverageMonthlySpend",
            "TenureGroup",
            "MonthlyChargeCategory",
            "ServiceEngagement",
            "SeniorCustomer",
        ]
        if column in data.columns
    ]

    print("\nEngineered features:")
    for feature in engineered_features:
        print(f"- {feature}")

    feature_summary = pd.DataFrame(
        {
            "Feature": engineered_features,
            "Description": [
                {
                    "CustomerLifetimeValue":
                        "Monthly charges multiplied by tenure.",
                    "AverageMonthlySpend":
                        "Average monthly spending based on total charges.",
                    "TenureGroup":
                        "Customer segment based on tenure.",
                    "MonthlyChargeCategory":
                        "Customer segment based on monthly charges.",
                    "ServiceEngagement":
                        "Count of subscribed yes/no services.",
                    "SeniorCustomer":
                        "Senior-customer category derived from SeniorCitizen.",
                }[feature]
                for feature in engineered_features
            ],
        }
    )
    save_dataframe(feature_summary, "feature_engineering_summary.csv")

    data.to_csv(
        OUTPUT_DIR / "feature_engineered_data.csv",
        index=False,
    )

    print(f"\nFeature-engineered dataset shape: {data.shape}")

    return data

def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build the complete train-only preprocessing + model pipeline.

    IQR capping, imputation, scaling and one-hot encoding are all fitted
    using X_train only.
    """

    numeric_features = [
        column
        for column in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[column])
    ]

    categorical_features = [
        column
        for column in X_train.columns
        if (
            pd.api.types.is_object_dtype(X_train[column])
            or pd.api.types.is_string_dtype(X_train[column])
            or pd.api.types.is_categorical_dtype(X_train[column])
            or pd.api.types.is_bool_dtype(X_train[column])
        )
    ]

    print("\nNumerical features:")
    for feature in numeric_features:
        print(f"- {feature}")

    print("\nCategorical features:")
    for feature in categorical_features:
        print(f"- {feature}")

    numeric_pipeline = Pipeline(
        steps=[
            # SimpleImputer returns a numeric array.
            ("imputer", SimpleImputer(strategy="median")),

            # IQR thresholds are learned ONLY from X_train.
            ("outlier_capper", IQRCapper(multiplier=1.5)),

            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline

def train_and_evaluate(data: pd.DataFrame):
    section("4. TRAIN-TEST SPLIT")

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    section("5. BUILDING COMPLETE PREPROCESSING + MODEL PIPELINE")

    pipeline = build_pipeline(X_train)
    print(pipeline)

    section("6. TRAINING")

    pipeline.fit(X_train, y_train)
    print("Pipeline trained successfully.")

    section("7. PREDICTION")

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    print(f"Predictions generated: {len(y_pred)}")

    section("8. MODEL EVALUATION")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    performance = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
            ],
        }
    )

    print(performance.to_string(index=False))
    save_dataframe(
        performance,
        "pipeline_model_performance.csv",
    )

    section("9. CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual_No_Churn", "Actual_Churn"],
        columns=["Predicted_No_Churn", "Predicted_Churn"],
    )
    confusion_df.to_csv(
        OUTPUT_DIR / "pipeline_confusion_matrix.csv"
    )

    # Save confusion-matrix visualization.
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])

    for row in range(2):
        for col in range(2):
            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
            )

    save_plot("pipeline_confusion_matrix.png")

    section("10. PREDICTION RESULTS")

    prediction_results = X_test.copy()
    prediction_results["Actual_Churn"] = y_test.values
    prediction_results["Predicted_Churn"] = y_pred
    prediction_results["Churn_Probability"] = y_probability

    save_dataframe(
        prediction_results.reset_index(drop=True),
        "pipeline_prediction_results.csv",
    )

    return (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probability,
        performance,
    )

def save_model_coefficients(pipeline: Pipeline) -> None:
    """
    Save logistic-regression coefficients for interpretability.

    This is a model-based interpretation, not a separate feature-selection
    training step. It avoids the leakage present in the original approach,
    where Random Forest feature selection was fitted before the train/test
    split.
    """

    section("11. MODEL FEATURE IMPORTANCE")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "Absolute_Coefficient": np.abs(coefficients),
            }
        ).sort_values(
            "Absolute_Coefficient",
            ascending=False,
        ).reset_index(drop=True)

        importance["Rank"] = (
            np.arange(len(importance)) + 1
        )

        importance = importance[
            [
                "Rank",
                "Feature",
                "Coefficient",
                "Absolute_Coefficient",
            ]
        ]

        save_dataframe(
            importance,
            "logistic_regression_feature_importance.csv",
        )

        print("\nTop 20 model coefficients:")
        print(
            importance.head(20).to_string(index=False)
        )

    except Exception as exc:
        print(f"Could not extract feature coefficients: {exc}")

def save_pipeline_and_report(
    pipeline: Pipeline,
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    section("12. SAVING FINAL PIPELINE")

    pipeline_file = (
        OUTPUT_DIR
        / "customer_churn_preprocessing_pipeline.pkl"
    )

    joblib.dump(pipeline, pipeline_file)
    print(f"Pipeline saved: {pipeline_file}")

    report_file = (
        OUTPUT_DIR / "preprocessing_pipeline_report.md"
    )

    report = f"""# Customer Churn Preprocessing & ML Pipeline

## Dataset
- Rows after cleaning: {len(data)}
- Features used: {data.shape[1] - 1}
- Training samples: {len(X_train)}
- Testing samples: {len(X_test)}

## Target
- Target column: `Churn`
- Encoding: No Churn = 0, Churn = 1

## Pipeline
1. Duplicate removal
2. CustomerID removal
3. Target cleaning
4. Deterministic feature engineering
5. Stratified 80/20 train-test split
6. Training-set IQR outlier capping
7. Training-set median imputation for numerical data
8. Standard scaling
9. Most-frequent imputation for categorical data
10. One-Hot Encoding
11. Balanced Logistic Regression

## Why This Approach Is Correct
All statistics learned from the data, including imputation values,
IQR outlier boundaries, scaling parameters and category mappings, are
learned through the pipeline using the training data only.

The test set is used only for final evaluation.

## Model Performance

{performance.to_string(index=False)}

## Output
The trained pipeline is saved as:

`{pipeline_file.name}`
"""

    report_file.write_text(
        report,
        encoding="utf-8",
    )

    print(f"Report saved: {report_file}")

def main() -> None:
    section("CUSTOMER CHURN - PREPROCESSING & MACHINE LEARNING")

    print(f"Base directory: {BASE_DIR}")
    print(f"Input file: {DATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Visualization directory: {VISUALIZATION_DIR}")

    raw_df = load_dataset()

    run_eda(raw_df)

    data = prepare_dataset(raw_df)

    (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probability,
        performance,
    ) = train_and_evaluate(data)

    save_model_coefficients(pipeline)

    save_pipeline_and_report(
        pipeline,
        data,
        X_train,
        X_test,
        performance,
    )

    section("PIPELINE COMPLETED SUCCESSFULLY")

    print("Final model: Logistic Regression")
    print("Preprocessing: sklearn Pipeline + ColumnTransformer")
    print("Data leakage protection: Enabled")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Visualizations: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
