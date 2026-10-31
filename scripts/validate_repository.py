"""Run lightweight, offline checks used by CI and local development."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import nbformat
import sklearn
from nbformat.validator import normalize


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_NOTEBOOKS = (
    ROOT / "rainfall_prediction.ipynb",
    ROOT / "FinalProject_AUSWeather_completed.ipynb",
    ROOT / "notebooks" / "tunisia_italy_rainfall_experiments.ipynb",
)
ARCHIVED_NOTEBOOK = ROOT / "original" / "FinalProject_AUSWeather_original.ipynb"
ARCHIVED_NOTEBOOK_SHA256 = (
    "dc9fb20e82585785b3fc9aecf0a4f43c2a8857643d9d3f25450a6be2d99c2373"
)
SCREENSHOT_SHA256 = {
    "australia-sample-predictions.png": (
        "f34c1c5928f77a5402a2c96e0ec62cf3aa3b42462eb1ac04d764a4661ca4adde"
    ),
    "open-meteo-download-sfax-rome.png": (
        "ddf255819a81ddc11d2e9db2780e817ddad8628cd30bab003a185e39c0ec7746"
    ),
    "sfax-rome-data-validation.png": (
        "05905237ad08be2951e1bab1e910ffb3f385742a8ca6c9390dc6c73c451a76c3"
    ),
    "sfax-rome-dataset-preview.png": (
        "b37fd6ea0c625bb2fdccf67b6e7eb22800dc2a55e4fd82da6ca8e0eeabbffe2a"
    ),
    "sfax-rome-model-comparison.png": (
        "809760967e3c14353a6b22ec66aa2d9d0c6f53daa80f049bc196d19622a89f4b"
    ),
    "sfax-rome-model-results.png": (
        "c8ead3a28edd5a3c158736d6f77c7df40be0436b91e2d090aa51a387d3d14ac1"
    ),
    "sfax-rome-sample-predictions.png": (
        "0cc3ddf1e9c39e6d14f63bf625d978261e5d88b0ddc04635c4eab9ddc3442fbc"
    ),
}
EXPECTED_FINAL_METRICS = {
    "Accuracy": 0.8412698412698413,
    "Precision": 0.7521367521367521,
    "Recall": 0.49162011173184356,
    "F1": 0.5945945945945946,
    "ROC-AUC": 0.885358674709294,
}
COLAB_URLS = (
    "https://colab.research.google.com/github/Melek-Lahmar/rainfall-prediction/blob/main/rainfall_prediction.ipynb",
    "https://colab.research.google.com/github/Melek-Lahmar/rainfall-prediction/blob/main/notebooks/tunisia_italy_rainfall_experiments.ipynb",
    "https://colab.research.google.com/github/Melek-Lahmar/rainfall-prediction/blob/main/FinalProject_AUSWeather_completed.ipynb",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_python() -> None:
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_notebooks() -> None:
    for path in MAINTAINED_NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        _, notebook = normalize(notebook)
        nbformat.validate(notebook)

        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path}:cell-{index}")
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        raise AssertionError(f"Saved notebook error in {path}:cell-{index}")

    if sha256(ARCHIVED_NOTEBOOK) != ARCHIVED_NOTEBOOK_SHA256:
        raise AssertionError("The byte-preserved IBM/Coursera notebook changed")


def validate_dependencies() -> None:
    version_match = re.match(r"^(\d+)\.(\d+)", sklearn.__version__)
    if not version_match:
        raise AssertionError(f"Unrecognized scikit-learn version: {sklearn.__version__}")
    version = tuple(map(int, version_match.groups()))
    if not (version >= (1, 8) and version < (1, 10)):
        raise AssertionError(
            "This project requires scikit-learn >=1.8,<1.10; "
            f"found {sklearn.__version__}"
        )

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if "scikit-learn>=1.8,<1.10" not in requirements:
        raise AssertionError("requirements.txt must preserve the reviewed scikit-learn range")


def validate_links_and_documentation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for url in COLAB_URLS:
        if url not in readme:
            raise AssertionError(f"Missing direct Colab link: {url}")

    for markdown_path in (ROOT / "README.md", ROOT / "data" / "README.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().split("#", maxsplit=1)[0]
            if not target or re.match(r"^(?:https?|mailto):", target):
                continue
            linked_path = (markdown_path.parent / target).resolve()
            if not linked_path.exists():
                raise AssertionError(f"Broken local link in {markdown_path}: {target}")

    cache_markers = (
        "weather_cache_partial.zip",
        "into `/content`",
        "requests only missing city-years",
        "Tunisie_Italie_2_Datasets_Meteo.zip",
    )
    for marker in cache_markers:
        if marker not in readme:
            raise AssertionError(f"README cache recovery text is missing: {marker}")


def validate_results_and_evidence() -> None:
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    if metrics["final_metrics"] != EXPECTED_FINAL_METRICS:
        raise AssertionError("Australian final metrics differ from the reviewed results")
    if metrics["split"]["chronological"] is not False:
        raise AssertionError("Australian split must remain documented as non-chronological")
    if metrics["source_notebook_sha256"] != ARCHIVED_NOTEBOOK_SHA256:
        raise AssertionError("Australian metric provenance hash is incorrect")

    screenshot_dir = ROOT / "assets" / "screenshots"
    actual_names = {path.name for path in screenshot_dir.glob("*.png")}
    if actual_names != set(SCREENSHOT_SHA256):
        raise AssertionError("The authentic seven-screenshot set changed")
    for name, expected_hash in SCREENSHOT_SHA256.items():
        if sha256(screenshot_dir / name) != expected_hash:
            raise AssertionError(f"Authentic screenshot changed: {name}")

    with (ROOT / "results" / "tunisia_italy_screenshot_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise AssertionError("Expected four screenshot-attributed Tunisia/Italy model rows")


def main() -> None:
    validate_python()
    validate_notebooks()
    validate_dependencies()
    validate_links_and_documentation()
    validate_results_and_evidence()
    print(
        "Offline validation passed: Python syntax, 3 maintained notebooks, "
        "dependencies, documentation links, metrics, and 7 authentic screenshots."
    )


if __name__ == "__main__":
    main()
