from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "weatherAUS_2.csv"
ORIGINAL_NOTEBOOK = ROOT / "original" / "FinalProject_AUSWeather_original.ipynb"
DOWNLOADED_NOTEBOOK = ROOT / "FinalProject_AUSWeather.ipynb"
SOURCE_NOTEBOOK = DOWNLOADED_NOTEBOOK if DOWNLOADED_NOTEBOOK.exists() else ORIGINAL_NOTEBOOK
PORTFOLIO_NOTEBOOK = ROOT / "rainfall_prediction.ipynb"
COURSE_NOTEBOOK = ROOT / "FinalProject_AUSWeather_completed.ipynb"
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
LOCATIONS = ["Melbourne", "MelbourneAirport", "Watsonia"]
PALETTE = {"No": "#4472C4", "Yes": "#ED7D31"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_original() -> None:
    ORIGINAL_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    if not ORIGINAL_NOTEBOOK.exists():
        shutil.copy2(SOURCE_NOTEBOOK, ORIGINAL_NOTEBOOK)
    if sha256(SOURCE_NOTEBOOK) != sha256(ORIGINAL_NOTEBOOK):
        raise RuntimeError("The preserved original notebook does not match the source notebook.")


def date_to_season(date: pd.Timestamp) -> str:
    if date.month in (12, 1, 2):
        return "Summer"
    if date.month in (3, 4, 5):
        return "Autumn"
    if date.month in (6, 7, 8):
        return "Winter"
    return "Spring"


def load_and_prepare_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_PATH)
    complete = raw.dropna().copy()
    local = complete.loc[complete["Location"].isin(LOCATIONS)].copy()
    local["Date"] = pd.to_datetime(local["Date"], errors="raise")
    local["Season"] = local["Date"].apply(date_to_season)
    local = local.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})
    local = local.drop(columns="Date")

    audit = {
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "raw_duplicates": int(raw.duplicated().sum()),
        "complete_rows": int(complete.shape[0]),
        "model_rows": int(local.shape[0]),
        "model_columns": int(local.shape[1]),
        "locations": LOCATIONS,
        "target_counts": {k: int(v) for k, v in local["RainToday"].value_counts().items()},
        "rain_rate": float((local["RainToday"] == "Yes").mean()),
    }
    return local, audit


def make_pipeline(classifier, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prediction = model.predict(X_test)
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
    yes_index = classes.index("Yes")
    probability = model.predict_proba(X_test)[:, yes_index]
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, prediction)),
        "Precision": float(precision_score(y_test, prediction, pos_label="Yes", zero_division=0)),
        "Recall": float(recall_score(y_test, prediction, pos_label="Yes", zero_division=0)),
        "F1": float(f1_score(y_test, prediction, pos_label="Yes", zero_division=0)),
        "ROC-AUC": float(roc_auc_score((y_test == "Yes").astype(int), probability)),
    }


def train_models(df: pd.DataFrame) -> dict:
    X = df.drop(columns="RainToday")
    y = df["RainToday"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    logistic_baseline = make_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        numeric_features,
        categorical_features,
    )
    logistic_baseline.fit(X_train, y_train)

    forest_baseline = make_pipeline(
        RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1),
        numeric_features,
        categorical_features,
    )
    forest_baseline.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logistic_search = GridSearchCV(
        make_pipeline(
            LogisticRegression(max_iter=3000, solver="saga", random_state=RANDOM_STATE),
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
    comparison = pd.DataFrame([evaluate(name, model, X_test, y_test) for name, model in models.items()])
    final_name = (
        "Random Forest (tuned)"
        if forest_search.best_score_ >= logistic_search.best_score_
        else "Logistic Regression (tuned)"
    )
    final_model = models[final_name]
    final_prediction = final_model.predict(X_test)
    final_classes = list(final_model.named_steps["classifier"].classes_)
    yes_index = final_classes.index("Yes")
    final_probability = final_model.predict_proba(X_test)[:, yes_index]

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cv": cv,
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
    bars = ax.bar(counts.index, counts.values, color=[PALETTE["No"], PALETTE["Yes"]])
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set(title="Rainfall class distribution", xlabel="Rain today", ylabel="Observations")
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    order = ["Summer", "Autumn", "Winter", "Spring"]
    seasonal = (
        df.assign(RainFlag=(df["RainToday"] == "Yes").astype(int))
        .groupby("Season", observed=False)["RainFlag"]
        .agg(["mean", "count"])
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(seasonal.index, seasonal["mean"] * 100, color="#4472C4")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in seasonal["mean"] * 100], padding=3)
    ax.set(title="Rainfall occurrence by season", xlabel="Season", ylabel="Rainy observations (%)")
    ax.set_ylim(0, (seasonal["mean"] * 100).max() * 1.18)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "seasonal_rainfall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison = trained["comparison"].set_index("Model")
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    comparison[metric_columns].plot(kind="bar", ax=ax, width=0.82, color=["#4472C4", "#A5A5A5", "#ED7D31", "#70AD47", "#8064A2"])
    ax.set(title="Model performance on the held-out test set", xlabel="", ylabel="Score", ylim=(0, 1.02))
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
        model = trained["models"][name]
        RocCurveDisplay.from_estimator(model, trained["X_test"], trained["y_test"], pos_label="Yes", name=name, ax=ax)
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
    importance = pd.DataFrame({"Feature": feature_names, label: values}).nlargest(20, label).sort_values(label)
    importance["Feature"] = importance["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance["Feature"], importance[label], color="#4472C4")
    ax.set(title=f"Top influential fitted features — {trained['final_name']}", xlabel=label, ylabel="")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_artifacts(audit: dict, trained: dict) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rainfall_classifier.joblib"
    joblib.dump(trained["final_model"], model_path)
    reloaded = joblib.load(model_path)
    sample = trained["X_test"].iloc[[0]]
    sample_prediction = reloaded.predict(sample)[0]
    sample_classes = list(reloaded.named_steps["classifier"].classes_)
    sample_probability = float(reloaded.predict_proba(sample)[0, sample_classes.index("Yes")])

    final_metrics = trained["comparison"].set_index("Model").loc[trained["final_name"]].drop(labels=[]).to_dict()
    final_metrics.pop("Model", None)
    payload = {
        "audit": audit,
        "split": {
            "train_rows": int(len(trained["X_train"])),
            "test_rows": int(len(trained["X_test"])),
            "random_state": RANDOM_STATE,
        },
        "models": trained["comparison"].to_dict(orient="records"),
        "grid_search": {
            "scoring": "roc_auc",
            "cv_folds": 5,
            "logistic_best_params": trained["logistic_search"].best_params_,
            "logistic_best_cv_score": float(trained["logistic_search"].best_score_),
            "forest_best_params": trained["forest_search"].best_params_,
            "forest_best_cv_score": float(trained["forest_search"].best_score_),
        },
        "final_model": trained["final_name"],
        "final_metrics": final_metrics,
        "confusion_matrix": confusion_matrix(
            trained["y_test"], trained["final_prediction"], labels=["No", "Yes"]
        ).tolist(),
        "classification_report": classification_report(
            trained["y_test"], trained["final_prediction"], output_dict=True, zero_division=0
        ),
        "example_prediction": {
            "test_index": str(sample.index[0]),
            "true_class": str(trained["y_test"].loc[sample.index[0]]),
            "predicted_class": str(sample_prediction),
            "rain_probability": sample_probability,
        },
        "model_reload_verified": bool(reloaded.predict(sample)[0] == sample_prediction),
        "source_notebook_sha256": sha256(SOURCE_NOTEBOOK),
        "preserved_notebook_sha256": sha256(ORIGINAL_NOTEBOOK),
    }
    payload = json_ready(payload)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trained["comparison"].to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return payload


def markdown_table(records: list[dict]) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['Model']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | "
            f"{row['Recall']:.3f} | {row['F1']:.3f} | {row['ROC-AUC']:.3f} |"
        )
    return "\n".join(lines)


def build_portfolio_notebook(payload: dict) -> None:
    fm = payload["final_metrics"]
    final_name = payload["final_model"]
    model_table = markdown_table(payload["models"])
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    cells = []
    m = nbf.v4.new_markdown_cell
    c = nbf.v4.new_code_cell

    cells.append(m(f"""# Australian Rainfall Prediction using Machine Learning

This project predicts rainfall for the Melbourne area using reproducible scikit-learn pipelines. It began as the IBM *Machine Learning with Python* final project and was expanded into a portfolio-ready analysis.

**Executed result:** `{final_name}` was selected by the strongest mean cross-validated ROC-AUC, without using the held-out test set for model selection. On the test set it achieved **{fm['Accuracy']:.3f} accuracy**, **{fm['Recall']:.3f} recall**, **{fm['F1']:.3f} F1**, and **{fm['ROC-AUC']:.3f} ROC-AUC**.

> All metrics and figures in this notebook are generated from `weatherAUS_2.csv`; no values are fabricated."""))
    cells.append(m("""## 1. Project Overview

The binary target is whether it rains on the current conceptual day. The original dataset's `RainTomorrow` is renamed `RainToday`, while original same-row weather measurements represent the preceding day. This preserves the IBM exercise's practical forecasting frame."""))
    cells.append(m("""## 2. Dataset

The file is the local `weatherAUS_2.csv`. The IBM notebook attributes the underlying observations to Australia's Bureau of Meteorology and the packaged dataset to Kaggle's *Rain in Australia* dataset. The analysis retains complete cases from Melbourne, Melbourne Airport, and Watsonia, matching the course workflow."""))
    cells.append(m("## 3. Imports and Configuration"))
    cells.append(c("""from pathlib import Path
import json
import joblib
import os
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("weatherAUS_2.csv")
LOCATIONS = ["Melbourne", "MelbourneAirport", "Watsonia"]
IMAGES_DIR = Path("images")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
for directory in (IMAGES_DIR, MODELS_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)

plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.18,
                     "axes.titleweight": "bold", "font.size": 10})"""))
    cells.append(m("## 4. Data Loading"))
    cells.append(c("""raw_df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")
raw_df.head(3)"""))
    cells.append(m("## 5. Data Understanding"))
    cells.append(c("""data_summary = pd.DataFrame({
    "dtype": raw_df.dtypes.astype(str),
    "missing": raw_df.isna().sum(),
    "unique": raw_df.nunique(dropna=True),
})
display(data_summary)
print("Duplicate rows:", raw_df.duplicated().sum())
print("Raw target values:", raw_df["RainTomorrow"].value_counts(dropna=False).to_dict())"""))
    cells.append(m("""## 6. Data Cleaning

To remain faithful to the IBM exercise, rows with any missing values are removed before restricting the geography. This complete-case choice reduces the available sample and may introduce selection bias; an imputer-based alternative is a reasonable future extension."""))
    cells.append(c("""complete_df = raw_df.dropna().copy()
df = complete_df.loc[complete_df["Location"].isin(LOCATIONS)].copy()
print(f"Complete rows across Australia: {len(complete_df):,}")
print(f"Complete rows in selected locations: {len(df):,}")
df["Location"].value_counts()"""))
    cells.append(m("""## 7. Data Leakage Considerations

Several measurements—such as rainfall, afternoon humidity, and cloud cover—are only known after the original observation day has progressed. Using them to predict that same original day's `RainToday` would leak information. Following the IBM notebook, the target is shifted conceptually by renaming original `RainTomorrow` to `RainToday` and original `RainToday` to `RainYesterday`. All remaining weather predictors are therefore interpreted as prior-day information relative to the target. Preprocessing is fitted inside each pipeline and only on training folds."""))
    cells.append(c("""df = df.rename(columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"})
assert "RainTomorrow" not in df.columns
df[["RainYesterday", "RainToday"]].head()"""))
    cells.append(m("## 8. Exploratory Data Analysis"))
    cells.append(c("""counts = df["RainToday"].value_counts().reindex(["No", "Yes"])
fig, ax = plt.subplots(figsize=(7.2, 4.5))
bars = ax.bar(counts.index, counts.values, color=["#4472C4", "#ED7D31"])
ax.bar_label(bars, fmt="{:,.0f}", padding=3)
ax.set(title="Rainfall class distribution", xlabel="Rain today", ylabel="Observations")
ax.set_ylim(0, counts.max() * 1.12)
fig.tight_layout()
fig.savefig(IMAGES_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
plt.show()
print(f"Rain occurs in {(df['RainToday'] == 'Yes').mean():.1%} of retained observations.")"""))
    cells.append(m("## 9. Feature Engineering"))
    cells.append(c("""def date_to_season(date):
    if date.month in (12, 1, 2): return "Summer"
    if date.month in (3, 4, 5): return "Autumn"
    if date.month in (6, 7, 8): return "Winter"
    return "Spring"

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["Season"] = df["Date"].apply(date_to_season)
df = df.drop(columns="Date")

season_order = ["Summer", "Autumn", "Winter", "Spring"]
seasonal = (df.assign(RainFlag=(df["RainToday"] == "Yes").astype(int))
              .groupby("Season", observed=False)["RainFlag"].agg(["mean", "count"])
              .reindex(season_order))
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(seasonal.index, seasonal["mean"] * 100, color="#4472C4")
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in seasonal["mean"] * 100], padding=3)
ax.set(title="Rainfall occurrence by season", xlabel="Season", ylabel="Rainy observations (%)")
ax.set_ylim(0, (seasonal["mean"] * 100).max() * 1.18)
fig.tight_layout()
fig.savefig(IMAGES_DIR / "seasonal_rainfall.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7.2, 4.8))
humidity_groups = [df.loc[df["RainToday"] == label, "Humidity3pm"] for label in ["No", "Yes"]]
ax.boxplot(humidity_groups, tick_labels=["No rain", "Rain"], patch_artist=True,
           boxprops={"facecolor": "#D9E5F6"}, medianprops={"color": "#ED7D31", "linewidth": 2})
ax.set(title="Prior-day afternoon humidity by rainfall outcome", xlabel="Rain today", ylabel="Humidity at 3pm (%)")
fig.tight_layout()
plt.show()"""))
    cells.append(m("## 10. Preprocessing"))
    cells.append(c("""X = df.drop(columns="RainToday")
y = df["RainToday"]
assert "RainToday" not in X.columns and "RainTomorrow" not in X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])
print(f"Train rows: {len(X_train):,}; test rows: {len(X_test):,}")
print(f"Numeric features: {len(numeric_features)}; categorical features: {len(categorical_features)}")"""))
    cells.append(m("## 11. Train/Test Split"))
    cells.append(c("""split_check = pd.DataFrame({
    "Full": y.value_counts(normalize=True),
    "Train": y_train.value_counts(normalize=True),
    "Test": y_test.value_counts(normalize=True),
})
display(split_check.round(3))"""))
    cells.append(m("## 12. Baseline Model"))
    cells.append(c("""def evaluate_model(name, model):
    prediction = model.predict(X_test)
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
    probability = model.predict_proba(X_test)[:, classes.index("Yes")]
    return {"Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, pos_label="Yes", zero_division=0),
            "Recall": recall_score(y_test, prediction, pos_label="Yes", zero_division=0),
            "F1": f1_score(y_test, prediction, pos_label="Yes", zero_division=0),
            "ROC-AUC": roc_auc_score((y_test == "Yes").astype(int), probability)}

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy_metrics = evaluate_model("Majority-class baseline", dummy)
pd.DataFrame([dummy_metrics]).round(3)"""))
    cells.append(m("## 13. Logistic Regression"))
    cells.append(c("""def make_pipeline(classifier):
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

logistic_baseline = make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
logistic_baseline.fit(X_train, y_train)
logistic_baseline_metrics = evaluate_model("Logistic Regression (baseline)", logistic_baseline)
pd.DataFrame([logistic_baseline_metrics]).round(3)"""))
    cells.append(m("## 14. Random Forest"))
    cells.append(c("""forest_baseline = make_pipeline(
    RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1))
forest_baseline.fit(X_train, y_train)
forest_baseline_metrics = evaluate_model("Random Forest (baseline)", forest_baseline)
pd.DataFrame([forest_baseline_metrics]).round(3)"""))
    cells.append(m("## 15. Hyperparameter Tuning"))
    cells.append(c("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logistic_search = GridSearchCV(
    make_pipeline(LogisticRegression(max_iter=3000, solver="saga", random_state=RANDOM_STATE)),
    {"classifier__C": [0.1, 1.0],
     "classifier__l1_ratio": [0.0, 1.0],
     "classifier__class_weight": [None, "balanced"]},
    scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
logistic_search.fit(X_train, y_train)

forest_search = GridSearchCV(
    make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    {"classifier__n_estimators": [150],
     "classifier__max_depth": [None, 10],
     "classifier__min_samples_split": [2, 5],
     "classifier__class_weight": [None, "balanced_subsample"]},
    scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
forest_search.fit(X_train, y_train)

print("Logistic best parameters:", logistic_search.best_params_)
print(f"Logistic best CV ROC-AUC: {logistic_search.best_score_:.4f}")
print("Forest best parameters:", forest_search.best_params_)
print(f"Forest best CV ROC-AUC: {forest_search.best_score_:.4f}")"""))
    cells.append(m("""## 16. Model Evaluation

Accuracy measures all correct decisions, but the minority rain class makes precision, recall, and F1 essential. Recall quantifies the share of rainy days detected; precision quantifies how often a rain prediction is correct; F1 balances the two. ROC-AUC evaluates ranking quality across thresholds."""))
    cells.append(c("""models = {
    "Majority-class baseline": dummy,
    "Logistic Regression (baseline)": logistic_baseline,
    "Random Forest (baseline)": forest_baseline,
    "Logistic Regression (tuned)": logistic_search.best_estimator_,
    "Random Forest (tuned)": forest_search.best_estimator_,
}
comparison = pd.DataFrame([evaluate_model(name, model) for name, model in models.items()])
display(comparison.round(3))

final_name = ("Random Forest (tuned)"
              if forest_search.best_score_ >= logistic_search.best_score_
              else "Logistic Regression (tuned)")
final_model = models[final_name]
final_prediction = final_model.predict(X_test)
final_classes = list(final_model.named_steps["classifier"].classes_)
final_probability = final_model.predict_proba(X_test)[:, final_classes.index("Yes")]
print("Selected model:", final_name)
print(classification_report(y_test, final_prediction, zero_division=0))"""))
    cells.append(m("## 17. Model Comparison"))
    cells.append(c("""metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(11, 6.2))
comparison.set_index("Model")[metric_columns].plot(
    kind="bar", ax=ax, width=0.82,
    color=["#4472C4", "#A5A5A5", "#ED7D31", "#70AD47", "#8064A2"])
ax.set(title="Model performance on the held-out test set", xlabel="", ylabel="Score", ylim=(0, 1.02))
ax.legend(loc="lower right", ncol=3, frameon=False)
ax.tick_params(axis="x", rotation=18)
fig.tight_layout()
fig.savefig(IMAGES_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(6.2, 5.2))
ConfusionMatrixDisplay.from_predictions(y_test, final_prediction, labels=["No", "Yes"],
    display_labels=["No rain", "Rain"], cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"Confusion matrix — {final_name}")
fig.tight_layout()
fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5.4))
for name in ["Logistic Regression (tuned)", "Random Forest (tuned)"]:
    RocCurveDisplay.from_estimator(models[name], X_test, y_test, pos_label="Yes", name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
ax.set_title("ROC curves on the held-out test set")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(IMAGES_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
plt.show()"""))
    cells.append(m("## 18. Feature Interpretation"))
    cells.append(c("""feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
classifier = final_model.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    values, value_label = classifier.feature_importances_, "Importance"
else:
    values, value_label = np.abs(classifier.coef_[0]), "Absolute coefficient"
importance = pd.DataFrame({"Feature": feature_names, value_label: values}).nlargest(20, value_label).sort_values(value_label)
importance["Feature"] = importance["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(importance["Feature"], importance[value_label], color="#4472C4")
ax.set(title=f"Top influential fitted features — {final_name}", xlabel=value_label, ylabel="")
fig.tight_layout()
fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
plt.show()
print("These features were influential for the fitted model; the ranking does not establish causality.")"""))
    cells.append(m("## 19. Final Model"))
    cells.append(c("""model_path = MODELS_DIR / "rainfall_classifier.joblib"
joblib.dump(final_model, model_path)
reloaded_model = joblib.load(model_path)
assert np.array_equal(reloaded_model.predict(X_test.iloc[:5]), final_model.predict(X_test.iloc[:5]))
print(f"Saved and reload-tested: {model_path}")"""))
    cells.append(m("## 20. Example Prediction"))
    cells.append(c("""sample = X_test.iloc[[0]]
sample_index = sample.index[0]
sample_prediction = reloaded_model.predict(sample)[0]
sample_classes = list(reloaded_model.named_steps["classifier"].classes_)
sample_probability = reloaded_model.predict_proba(sample)[0, sample_classes.index("Yes")]
pd.DataFrame({"True class": [y_test.loc[sample_index]],
              "Predicted class": [sample_prediction],
              "Rain probability": [sample_probability]}, index=[sample_index])"""))
    cells.append(m("""## 21. Limitations

- Complete-case filtering discards many rows and can bias the retained sample.
- The scope is limited to three nearby Melbourne-area stations.
- A random stratified split supports the IBM exercise but can overstate deployment performance when nearby dates are correlated; time-aware validation is a valuable next step.
- Feature influence is predictive, not causal.
- Weather and climate patterns may shift beyond the 2007–2017 observation period."""))
    cells.append(m(f"""## 22. Conclusion

The selected **{final_name}** achieved **{fm['Accuracy']:.3f} accuracy**, **{fm['Precision']:.3f} precision**, **{fm['Recall']:.3f} recall**, **{fm['F1']:.3f} F1**, and **{fm['ROC-AUC']:.3f} ROC-AUC** on the held-out test set. The complete fitted preprocessing-and-classifier pipeline is saved to `models/rainfall_classifier.joblib`.

### Executed model comparison

{model_table}"""))
    nb["cells"] = cells
    nbf.write(nb, PORTFOLIO_NOTEBOOK)


def build_course_notebook() -> None:
    nb = nbf.read(SOURCE_NOTEBOOK, as_version=4)
    nb["nbformat_minor"] = 5
    # Normalize malformed legacy cell fields only in the completed copy.
    # The preserved original remains byte-for-byte unchanged.
    for index, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"cell-{index:03d}")
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    nb["cells"][4]["source"] = "print('Dependencies are managed by requirements.txt; no runtime installation is needed.')"
    nb["cells"][5]["source"] = """import os
os.environ.setdefault('MPLCONFIGDIR', '.matplotlib')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore', message="'penalty' was deprecated", category=FutureWarning)
warnings.filterwarnings('ignore', message='Inconsistent values: penalty=', category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, recall_score)
RANDOM_STATE = 42"""
    nb["cells"][8]["source"] = """df = pd.read_csv('weatherAUS_2.csv')
df.head()"""
    nb["cells"][27]["source"] = """# Exercise 1
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date')
df.head()"""
    nb["cells"][30]["source"] = """# Exercise 2
X = df.drop(columns='RainToday')
y = df['RainToday']
assert 'RainToday' not in X.columns and 'RainTomorrow' not in X.columns"""
    nb["cells"][32]["source"] = """# Exercise 3
y.value_counts()"""
    nb["cells"][34] = nbf.v4.new_markdown_cell("""### Exercise 4 answer

- Rain occurs in about 24% of the retained Melbourne-area observations (roughly 87 days per 365-day year if the sample rate is treated as an annual approximation).
- Always predicting “No” would be about 76% accurate.
- The target is imbalanced because the no-rain class is substantially larger.
- Stratified splitting, class-sensitive metrics, and model comparison are therefore important.""")
    nb["cells"][36]["source"] = """# Exercise 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)"""
    nb["cells"][38]["source"] = """# Exercise 6
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
numeric_features, categorical_features"""
    nb["cells"][42]["source"] = """# Exercise 7
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])"""
    nb["cells"][44]["source"] = """# Exercise 8
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])"""
    nb["cells"][49]["source"] = "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)"
    nb["cells"][51]["source"] = """# Exercise 9
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)"""
    nb["cells"][55]["source"] = """# Exercise 10
test_score = grid_search.score(X_test, y_test)
print('Test set score: {:.4f}'.format(test_score))"""
    nb["cells"][58]["source"] = """# Exercise 11
y_pred = grid_search.predict(X_test)"""
    nb["cells"][60]["source"] = """# Exercise 12
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))"""
    nb["cells"][62]["source"] = """# Exercise 13
conf_matrix = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No rain', 'Rain'])
disp.plot(cmap='Blues', colorbar=False)
plt.title('Random Forest Confusion Matrix')
plt.show()
rf_accuracy = accuracy_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred, pos_label='Yes')
print(f'Random Forest accuracy: {rf_accuracy:.4f}')
print(f'Random Forest true positive rate (recall): {rf_recall:.4f}')"""
    nb["cells"][67]["source"] = """# Exercise 14
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_
feature_importances[:5]"""
    nb["cells"][72]["source"] = """# Exercise 15
pipeline.set_params(classifier=LogisticRegression(random_state=RANDOM_STATE, max_iter=2000))
logistic_param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}
model = GridSearchCV(pipeline, logistic_param_grid, cv=cv, scoring='accuracy', n_jobs=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Best Logistic Regression parameters:', model.best_params_)"""
    nb["cells"][74]["source"] = """print(classification_report(y_test, y_pred, zero_division=0))
conf_matrix = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No rain', 'Rain'])
disp.plot(cmap='Blues', colorbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.show()
lr_accuracy = accuracy_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred, pos_label='Yes')
print(f'Logistic Regression accuracy: {lr_accuracy:.4f}')
print(f'Logistic Regression true positive rate (recall): {lr_recall:.4f}')"""
    nb["cells"][75] = nbf.v4.new_markdown_cell("""### Model comparison conclusion

The executed classification reports and confusion matrices above provide the course comparison. Accuracy measures overall correctness, while the true-positive rate is the recall reported for the `Yes` class. The models should be compared on both because the target is imbalanced.""")
    nbf.write(nb, COURSE_NOTEBOOK)


def write_text_files(payload: dict) -> None:
    fm = payload["final_metrics"]
    grid = payload["grid_search"]
    final_name = payload["final_model"]
    model_table = markdown_table(payload["models"])
    best_params = grid["logistic_best_params"] if "Logistic" in final_name else grid["forest_best_params"]
    best_cv = grid["logistic_best_cv_score"] if "Logistic" in final_name else grid["forest_best_cv_score"]
    cm = payload["confusion_matrix"]

    readme = f"""# 🌧️ Australian Rainfall Prediction using Machine Learning

## Overview

An end-to-end binary classification project that predicts rainfall for the Melbourne area from historical Australian weather observations. The repository contains an executed notebook, leakage-aware preprocessing pipelines, cross-validated model tuning, reproducible figures, and a serialized final pipeline.

Repository: [Melek-Lahmar/australian-rainfall-prediction](https://github.com/Melek-Lahmar/australian-rainfall-prediction)

## Project Context

The project originated from IBM's *Machine Learning with Python* coursework on Coursera and was independently enhanced into a portfolio project. IBM does not endorse this enhanced repository.

## Objective

Predict whether it rains on the current conceptual day using weather measurements available through the preceding day.

## Dataset

- Local file: `weatherAUS_2.csv` (intentionally ignored by Git while redistribution terms remain unclear)
- Source stated by the IBM notebook: Australian Bureau of Meteorology observations, packaged as Kaggle's *Rain in Australia* dataset
- Raw shape: **{payload['audit']['raw_rows']:,} rows × {payload['audit']['raw_columns']} columns**
- Modeling sample: **{payload['audit']['model_rows']:,} complete observations**
- Geographic scope: Melbourne, Melbourne Airport, and Watsonia
- Target: `RainToday` after the IBM target-shift convention
- Positive-class rate: **{payload['audit']['rain_rate']:.1%}**

See [data/README.md](data/README.md) for source and setup details.

## Machine Learning Workflow

Data Loading → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Cross-Validation → GridSearchCV → Evaluation → Final Pipeline

## Technologies

Python, NumPy, Pandas, Matplotlib, scikit-learn, Jupyter, and joblib.

## Exploratory Data Analysis

![Class distribution](images/class_distribution.png)

![Seasonal rainfall](images/seasonal_rainfall.png)

## Data Leakage Considerations

The raw file contains `RainToday` and `RainTomorrow`. Same-day weather measurements can leak information if used to predict the original `RainToday`. Following the IBM exercise, original `RainTomorrow` becomes the modeling target `RainToday`, original `RainToday` becomes `RainYesterday`, and all other same-row weather measurements are interpreted as prior-day information. Preprocessing is fitted only inside training and cross-validation pipelines.

## Preprocessing

The course workflow removes incomplete rows. Numerical predictors are standardized, categorical predictors are one-hot encoded with unknown-category handling, and both transformations are fitted inside a `ColumnTransformer`/`Pipeline` to prevent preprocessing leakage.

## Models

- Majority-class baseline
- Logistic Regression
- Random Forest

Baseline and tuned versions of both classifiers are evaluated.

## Hyperparameter Tuning

Both classifiers use five-fold `StratifiedKFold` cross-validation. `GridSearchCV` optimizes ROC-AUC over compact, model-specific parameter grids. The selected model's best parameters are `{best_params}` and its best mean cross-validation ROC-AUC is **{best_cv:.4f}**.

## Results

{model_table}

The selected model is **{final_name}**. Its held-out confusion matrix is `[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]` for actual/predicted class order `No, Yes`.

## Model Comparison

![Model comparison](images/model_comparison.png)

## Confusion Matrix

![Confusion matrix](images/confusion_matrix.png)

## ROC Curve

![ROC curve](images/roc_curve.png)

## Feature Importance

![Feature influence](images/feature_importance.png)

The plot describes influence in the fitted model; it does not establish causality.

## Key Findings

- The retained target is imbalanced: rain occurs in **{payload['audit']['rain_rate']:.1%}** of observations, so accuracy alone is insufficient.
- **{final_name}** had the strongest mean cross-validated ROC-AUC among the tuned candidates and was selected before inspecting final test metrics.
- Tuned Logistic Regression detected more rainy days on the test set, while the selected Random Forest provided materially higher precision and ROC-AUC; threshold choice should follow the operational cost of missed rain versus false alarms.
- Final held-out metrics are **{fm['Accuracy']:.3f} accuracy**, **{fm['Precision']:.3f} precision**, **{fm['Recall']:.3f} recall**, **{fm['F1']:.3f} F1**, and **{fm['ROC-AUC']:.3f} ROC-AUC**.
- The saved artifact contains the complete preprocessing-and-classifier pipeline and passed a reload prediction test.

## Project Structure

```text
.
├── README.md
├── rainfall_prediction.ipynb
├── FinalProject_AUSWeather_completed.ipynb
├── requirements.txt
├── .gitignore
├── LINKEDIN_POST.md
├── original/
│   └── FinalProject_AUSWeather_original.ipynb
├── data/
│   └── README.md
├── images/
│   ├── class_distribution.png
│   ├── seasonal_rainfall.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── models/
│   └── rainfall_classifier.joblib
├── results/
│   ├── metrics.json
│   └── model_comparison.csv
└── scripts/
    └── build_project.py
```

`weatherAUS_2.csv` remains at repository root for local execution but is not included in the proposed Git commit.

## Installation

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

Place `weatherAUS_2.csv` at the repository root before running the notebooks.

## Usage

```bash
jupyter notebook rainfall_prediction.ipynb
```

Run all cells from top to bottom. For the IBM/Coursera course submission, use `FinalProject_AUSWeather_completed.ipynb` after reviewing the exercise answers.

## Reproducing the Results

```bash
python scripts/build_project.py
python -m jupyter nbconvert --execute --to notebook --inplace rainfall_prediction.ipynb
```

The scripts and notebook use `RANDOM_STATE = 42`. Small numerical differences may occur across library versions or hardware.

## Limitations

- Complete-case filtering discards many observations and may introduce selection bias.
- The model is localized to three Melbourne-area stations.
- Random stratified validation can be optimistic for temporally correlated weather; a time-aware holdout should be evaluated before operational use.
- The source period ends in 2017, and climate/weather relationships may drift.

## Future Improvements

- Compare complete-case results with train-only imputation.
- Add time-aware validation and probability calibration.
- Evaluate broader geographic generalization.
- Refresh with newer Bureau of Meteorology observations where licensing and schema permit.

## Course Context

Based on the IBM *Machine Learning with Python* final project on Coursera. The original notebook is preserved unchanged under `original/`.

## Author

Melek Lahmar
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    linkedin = f"""# LinkedIn Post

I completed an end-to-end Australian rainfall prediction project as part of IBM's *Machine Learning with Python* coursework and expanded it into a reproducible portfolio repository.

The objective was to classify rainfall for the Melbourne area using historical weather observations while paying close attention to target definition and data leakage.

I implemented:

- data cleaning and season feature engineering;
- leakage-aware `ColumnTransformer` and scikit-learn pipelines;
- Logistic Regression and Random Forest baselines;
- stratified cross-validation and `GridSearchCV`;
- evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves;
- model interpretation and serialization of the complete fitted pipeline.

The selected **{final_name}** achieved **{fm['Accuracy']:.3f} accuracy**, **{fm['Precision']:.3f} precision**, **{fm['Recall']:.3f} recall**, **{fm['F1']:.3f} F1**, and **{fm['ROC-AUC']:.3f} ROC-AUC** on the held-out test set.

The biggest lesson was that a strong classification project is not just about accuracy: defining the prediction time correctly, preventing leakage, and evaluating the minority class are equally important.

Technologies: Python, Pandas, NumPy, Matplotlib, scikit-learn, Jupyter, and joblib.

GitHub: https://github.com/Melek-Lahmar/australian-rainfall-prediction

#MachineLearning #Python #DataScience #ScikitLearn #IBM #ArtificialIntelligence

## LinkedIn Projects Section Description

Built and validated a leakage-aware rainfall classifier for the Melbourne area using scikit-learn pipelines, stratified cross-validation, and GridSearchCV. Compared Logistic Regression and Random Forest across accuracy, precision, recall, F1, and ROC-AUC, then saved the complete final pipeline for reproducible inference.

## Credential

[Add IBM/Coursera credential URL after course completion]
"""
    (ROOT / "LINKEDIN_POST.md").write_text(linkedin, encoding="utf-8")

    (ROOT / "data").mkdir(exist_ok=True)
    data_readme = f"""# Dataset Notes

## Dataset

- Local filename: `weatherAUS_2.csv`
- Raw shape: {payload['audit']['raw_rows']:,} rows × {payload['audit']['raw_columns']} columns
- Observation period in the local file: 2007-11-01 through 2017-06-25
- Rows represent daily weather observations at Australian weather stations.
- Modeling target: original `RainTomorrow`, renamed `RainToday` under the IBM course's prediction-time convention.

## Source

The original IBM notebook attributes the observations to the [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) and states that the course dataset was obtained from Kaggle's [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/) package.

## Modeling Scope

The course workflow drops rows containing missing values and retains Melbourne, Melbourne Airport, and Watsonia. This leaves {payload['audit']['model_rows']:,} observations.

## Local Setup

Place `weatherAUS_2.csv` in the repository root. The notebook loads it with:

```python
pd.read_csv("weatherAUS_2.csv")
```

The CSV is ignored by Git because the redistribution/license terms for the packaged course copy were not verified. Obtain the dataset through the course materials or the source links above and review their current terms before redistributing it.
"""
    (ROOT / "data" / "README.md").write_text(data_readme, encoding="utf-8")


def main() -> None:
    preserve_original()
    df, audit = load_and_prepare_data()
    trained = train_models(df)
    save_figures(df, trained)
    payload = save_artifacts(audit, trained)
    build_portfolio_notebook(payload)
    build_course_notebook()
    write_text_files(payload)
    print(json.dumps({
        "final_model": payload["final_model"],
        "final_metrics": payload["final_metrics"],
        "best_logistic": payload["grid_search"]["logistic_best_params"],
        "best_forest": payload["grid_search"]["forest_best_params"],
    }, indent=2))


if __name__ == "__main__":
    main()
