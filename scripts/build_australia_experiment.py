"""Rebuild the Australian experiment's model, metrics, and figures.

This script intentionally does not rewrite project documentation or notebooks.
The original IBM/Coursera notebook under ``original/`` is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402


DEFAULT_DATA_PATH = ROOT / "weatherAUS_2.csv"
ORIGINAL_NOTEBOOK = ROOT / "original" / "FinalProject_AUSWeather_original.ipynb"
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
LOCATIONS = ["Melbourne", "MelbourneAirport", "Watsonia"]
PALETTE = {"No": "#4472C4", "Yes": "#ED7D31"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the 23-column Australian weather CSV.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_to_season(date: pd.Timestamp) -> str:
    """Map an Australian calendar date to its meteorological season."""
    if date.month in (12, 1, 2):
        return "Summer"
    if date.month in (3, 4, 5):
        return "Autumn"
    if date.month in (6, 7, 8):
        return "Winter"
    return "Spring"


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, dict]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. See data/README.md for setup."
        )

    raw = pd.read_csv(data_path)
    expected_columns = {
        "Date",
        "Location",
        "RainToday",
        "RainTomorrow",
    }
    missing_columns = expected_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    complete = raw.dropna().copy()
    local = complete.loc[complete["Location"].isin(LOCATIONS)].copy()
    local["Date"] = pd.to_datetime(local["Date"], errors="raise")
    local["Season"] = local["Date"].apply(date_to_season)

    # The IBM exercise predicts the original next-day label. These names make
    # the timing explicit while retaining the course workflow.
    local = local.rename(
        columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"}
    )
    local = local.drop(columns="Date")

    if local.empty:
        raise ValueError("No complete rows remain for the configured locations.")
    if set(local["RainToday"].unique()) != {"No", "Yes"}:
        raise ValueError("RainToday must contain exactly the labels 'No' and 'Yes'.")

    audit = {
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "raw_duplicates": int(raw.duplicated().sum()),
        "complete_rows": int(complete.shape[0]),
        "model_rows": int(local.shape[0]),
        "model_columns": int(local.shape[1]),
        "locations": LOCATIONS,
        "target_counts": {
            str(label): int(count)
            for label, count in local["RainToday"].value_counts().items()
        },
        "rain_rate": float((local["RainToday"] == "Yes").mean()),
    }
    return local, audit


def make_pipeline(
    classifier: object,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def evaluate(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    prediction = model.predict(X_test)
    classes = list(model.classes_)
    yes_index = classes.index("Yes")
    probability = model.predict_proba(X_test)[:, yes_index]
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, prediction)),
        "Precision": float(
            precision_score(y_test, prediction, pos_label="Yes", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, prediction, pos_label="Yes", zero_division=0)
        ),
        "F1": float(f1_score(y_test, prediction, pos_label="Yes", zero_division=0)),
        "ROC-AUC": float(
            roc_auc_score((y_test == "Yes").astype(int), probability)
        ),
    }


def train_models(df: pd.DataFrame) -> dict:
    X = df.drop(columns="RainToday")
    y = df["RainToday"]
    if "RainToday" in X.columns or "RainTomorrow" in X.columns:
        raise AssertionError("Target leakage detected in the predictor matrix.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    logistic_baseline = make_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        numeric_features,
        categorical_features,
    )
    logistic_baseline.fit(X_train, y_train)

    forest_baseline = make_pipeline(
        RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        numeric_features,
        categorical_features,
    )
    forest_baseline.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logistic_search = GridSearchCV(
        make_pipeline(
            LogisticRegression(
                max_iter=3000,
                solver="saga",
                random_state=RANDOM_STATE,
            ),
            numeric_features,
            categorical_features,
        ),
        {
            "classifier__C": [0.1, 1.0],
            "classifier__l1_ratio": [0.0, 1.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    logistic_search.fit(X_train, y_train)

    forest_search = GridSearchCV(
        make_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            numeric_features,
            categorical_features,
        ),
        {
            "classifier__n_estimators": [150],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": [None, "balanced_subsample"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    forest_search.fit(X_train, y_train)

    models = {
        "Majority-class baseline": dummy,
        "Logistic Regression (baseline)": logistic_baseline,
        "Random Forest (baseline)": forest_baseline,
        "Logistic Regression (tuned)": logistic_search.best_estimator_,
        "Random Forest (tuned)": forest_search.best_estimator_,
    }
    comparison = pd.DataFrame(
        [evaluate(name, model, X_test, y_test) for name, model in models.items()]
    )
    final_name = (
        "Random Forest (tuned)"
        if forest_search.best_score_ >= logistic_search.best_score_
        else "Logistic Regression (tuned)"
    )
    final_model = models[final_name]
    final_prediction = final_model.predict(X_test)
    final_classes = list(final_model.classes_)
    final_probability = final_model.predict_proba(X_test)[
        :, final_classes.index("Yes")
    ]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "comparison": comparison,
        "logistic_search": logistic_search,
        "forest_search": forest_search,
        "final_name": final_name,
        "final_model": final_model,
        "final_prediction": final_prediction,
        "final_probability": final_probability,
    }


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
        }
    )


def save_figures(df: pd.DataFrame, trained: dict) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    counts = df["RainToday"].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[PALETTE["No"], PALETTE["Yes"]],
    )
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set(
        title="Rainfall class distribution",
        xlabel="Next-day rain label",
        ylabel="Observations",
    )
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    order = ["Summer", "Autumn", "Winter", "Spring"]
    seasonal = (
        df.assign(RainFlag=(df["RainToday"] == "Yes").astype(int))
        .groupby("Season", observed=False)["RainFlag"]
        .mean()
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(seasonal.index, seasonal * 100, color="#4472C4")
    ax.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in seasonal * 100],
        padding=3,
    )
    ax.set(
        title="Next-day rainfall occurrence by season",
        xlabel="Season",
        ylabel="Rainy labels (%)",
    )
    ax.set_ylim(0, (seasonal * 100).max() * 1.18)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "seasonal_rainfall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison = trained["comparison"].set_index("Model")
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    comparison[metric_columns].plot(
        kind="bar",
        ax=ax,
        width=0.82,
        color=["#4472C4", "#A5A5A5", "#ED7D31", "#70AD47", "#8064A2"],
    )
    ax.set(
        title="Model performance on the held-out test set",
        xlabel="",
        ylabel="Score",
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", ncol=3, frameon=False)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        trained["y_test"],
        trained["final_prediction"],
        labels=["No", "Yes"],
        display_labels=["No rain", "Rain"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {trained['final_name']}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    for name in ["Logistic Regression (tuned)", "Random Forest (tuned)"]:
        RocCurveDisplay.from_estimator(
            trained["models"][name],
            trained["X_test"],
            trained["y_test"],
            pos_label="Yes",
            name=name,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_model = trained["final_model"]
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    classifier = final_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        label = "Importance"
    else:
        values = np.abs(classifier.coef_[0])
        label = "Absolute coefficient"
    importance = (
        pd.DataFrame({"Feature": feature_names, label: values})
        .nlargest(20, label)
        .sort_values(label)
    )
    importance["Feature"] = importance["Feature"].str.replace(
        r"^(num|cat)__", "", regex=True
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance["Feature"], importance[label], color="#4472C4")
    ax.set(
        title=f"Top influential fitted features — {trained['final_name']}",
        xlabel=label,
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def to_json_ready(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    return value


def save_artifacts(audit: dict, trained: dict) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "rainfall_classifier.joblib"
    joblib.dump(trained["final_model"], model_path)
    reloaded = joblib.load(model_path)
    sample = trained["X_test"].iloc[[0]]
    sample_prediction = reloaded.predict(sample)[0]
    sample_classes = list(reloaded.classes_)
    sample_probability = float(
        reloaded.predict_proba(sample)[0, sample_classes.index("Yes")]
    )

    comparison = trained["comparison"].set_index("Model")
    final_metrics = comparison.loc[trained["final_name"]].to_dict()
    payload = {
        "audit": audit,
        "split": {
            "design": "random_stratified_holdout",
            "train_rows": int(len(trained["X_train"])),
            "test_rows": int(len(trained["X_test"])),
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "chronological": False,
        },
        "models": trained["comparison"].to_dict(orient="records"),
        "grid_search": {
            "scoring": "roc_auc",
            "cv_folds": 5,
            "cv_strategy": "StratifiedKFold(shuffle=True, random_state=42)",
            "logistic_best_params": trained["logistic_search"].best_params_,
            "logistic_best_cv_score": float(
                trained["logistic_search"].best_score_
            ),
            "forest_best_params": trained["forest_search"].best_params_,
            "forest_best_cv_score": float(trained["forest_search"].best_score_),
        },
        "final_model": trained["final_name"],
        "final_metrics": final_metrics,
        "confusion_matrix": confusion_matrix(
            trained["y_test"],
            trained["final_prediction"],
            labels=["No", "Yes"],
        ).tolist(),
        "classification_report": classification_report(
            trained["y_test"],
            trained["final_prediction"],
            output_dict=True,
            zero_division=0,
        ),
        "example_prediction": {
            "test_index": str(sample.index[0]),
            "true_class": str(trained["y_test"].loc[sample.index[0]]),
            "predicted_class": str(sample_prediction),
            "rain_probability": sample_probability,
        },
        "model_reload_verified": bool(
            reloaded.predict(sample)[0] == sample_prediction
        ),
        "source_notebook_sha256": sha256(ORIGINAL_NOTEBOOK),
        "preserved_notebook_sha256": sha256(ORIGINAL_NOTEBOOK),
    }
    payload = to_json_ready(payload)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    trained["comparison"].to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )
    return payload


def main() -> None:
    args = parse_args()
    df, audit = load_and_prepare_data(args.data.resolve())
    trained = train_models(df)
    save_figures(df, trained)
    payload = save_artifacts(audit, trained)
    print(
        json.dumps(
            {
                "final_model": payload["final_model"],
                "final_metrics": payload["final_metrics"],
                "evaluation_design": payload["split"]["design"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
