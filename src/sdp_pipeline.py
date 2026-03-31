from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
from xgboost import XGBClassifier

try:
    import cupy as cp
except ImportError:  # optional dependency for end-to-end GPU XGBoost
    cp = None

TARGET_COLUMN = "defective"
TEXT_FALLBACK_COLUMN = "sd"
TARGET_SOURCE_COLUMN = "rs"
DEFAULT_SEPARATOR = ";"
LEAKAGE_COLUMNS = {TARGET_SOURCE_COLUMN}
NON_FEATURE_COLUMNS = {"bugID", TARGET_COLUMN}

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from .*")
warnings.filterwarnings(
    "ignore",
    message=r"You are sending unauthenticated requests to the HF Hub.*",
)
hf_logging.set_verbosity_error()


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_xgboost_device() -> str:
    override = os.getenv("SDP_XGB_DEVICE")
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_xgboost_ready() -> bool:
    return get_xgboost_device().startswith("cuda") and cp is not None


def load_dataset(csv_path: str | Path, sep: str = DEFAULT_SEPARATOR) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep)
    required_columns = {TARGET_SOURCE_COLUMN, TEXT_FALLBACK_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df = df.dropna(subset=[TARGET_SOURCE_COLUMN]).reset_index(drop=True)
    df[TEXT_FALLBACK_COLUMN] = df[TEXT_FALLBACK_COLUMN].fillna("")
    df[TARGET_COLUMN] = (
        df[TARGET_SOURCE_COLUMN].astype(str).str.upper().eq("FIXED").astype(int)
    )
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_candidate_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS and col not in LEAKAGE_COLUMNS
    ]


def _original_column_for_encoded_feature(
    encoded_feature: str, original_columns: Iterable[str]
) -> str:
    for column in sorted(original_columns, key=len, reverse=True):
        if encoded_feature == column or encoded_feature.startswith(f"{column}_"):
            return column
    return encoded_feature.split("_", 1)[0]


def aggregate_feature_scores(
    score_series: pd.Series,
    original_columns: Iterable[str],
    top_n_features: int = 30,
    final_n_columns: int = 5,
    default_col: str = TEXT_FALLBACK_COLUMN,
) -> Tuple[List[str], pd.DataFrame]:
    top = score_series.head(top_n_features).reset_index()
    top.columns = ["encoded_feature", "score"]
    top["original_col"] = top["encoded_feature"].apply(
        lambda value: _original_column_for_encoded_feature(value, original_columns)
    )

    agg = (
        top.groupby("original_col")
        .agg(max_score=("score", "max"), frequency=("score", "count"))
        .reset_index()
        .sort_values(by=["max_score", "frequency"], ascending=[False, False])
    )

    best_columns = [default_col]
    additional_cols = (
        agg[agg["original_col"] != default_col]
        .head(max(final_n_columns - 1, 0))["original_col"]
        .tolist()
    )
    best_columns.extend(additional_cols)
    return best_columns, agg


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x.astype(str), y)
    chi2_value = ss.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2_value / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denominator))


def rank_feature_groups(
    df: pd.DataFrame,
    final_n_columns: int = 5,
    top_n_features: int = 30,
) -> Dict[str, Dict[str, object]]:
    feature_columns = get_candidate_feature_columns(df)
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN]
    X_encoded = pd.get_dummies(X.astype(str), dummy_na=False)

    rankings: Dict[str, Dict[str, object]] = {}

    mi_scores = pd.Series(
        mutual_info_classif(X_encoded, y, discrete_features=True, random_state=42),
        index=X_encoded.columns,
    ).sort_values(ascending=False)
    best, aggregated = aggregate_feature_scores(
        mi_scores, feature_columns, top_n_features, final_n_columns
    )
    rankings["mutual_info"] = {"scores": mi_scores, "best_columns": best, "table": aggregated}

    chi_scores, _ = chi2(X_encoded, y)
    chi_series = pd.Series(chi_scores, index=X_encoded.columns).sort_values(
        ascending=False
    )
    best, aggregated = aggregate_feature_scores(
        chi_series, feature_columns, top_n_features, final_n_columns
    )
    rankings["chi_square"] = {
        "scores": chi_series,
        "best_columns": best,
        "table": aggregated,
    }

    cramers_scores = pd.Series(
        {column: cramers_v(X[column], y) for column in feature_columns}
    ).sort_values(ascending=False)
    best, aggregated = aggregate_feature_scores(
        cramers_scores, feature_columns, top_n_features, final_n_columns
    )
    rankings["cramers_v"] = {
        "scores": cramers_scores,
        "best_columns": best,
        "table": aggregated,
    }

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)
    rf_scores = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(
        ascending=False
    )
    best, aggregated = aggregate_feature_scores(
        rf_scores, feature_columns, top_n_features, final_n_columns
    )
    rankings["random_forest"] = {
        "scores": rf_scores,
        "best_columns": best,
        "table": aggregated,
    }

    return rankings


def build_intent_text(df: pd.DataFrame, selected_columns: List[str]) -> pd.Series:
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Selected columns missing from input: {missing_columns}")

    intent = (
        df[selected_columns]
        .fillna("")
        .astype(str)
        .apply(lambda row: " ".join(part.strip() for part in row if part.strip()), axis=1)
    )
    return intent.str.replace(r"\s+", " ", regex=True).str.strip()


def build_model_inputs(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    prepared = df.copy()
    prepared["intent_text"] = build_intent_text(prepared, selected_columns)
    prepared = prepared[prepared["intent_text"].ne("")].copy()
    prepared = prepared.drop_duplicates(subset=["intent_text"]).reset_index(drop=True)
    return prepared


@lru_cache(maxsize=4)
def _load_transformer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    device = get_torch_device()
    return tokenizer, model.to(device), device


def get_embeddings(
    texts: List[str],
    model_name: str = "roberta-base",
    batch_size: int = 32,
    max_length: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    tokenizer, model, device = _load_transformer(model_name)
    embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding text")

    for index in iterator:
        batch = texts[index : index + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embeddings)

    if not embeddings:
        return np.empty((0, 768))
    return np.vstack(embeddings)


def get_embeddings_for_xgboost(
    texts: List[str],
    model_name: str = "roberta-base",
    batch_size: int = 32,
    max_length: int = 256,
    gpu_mode: bool | None = None,
    show_progress: bool = True,
):
    tokenizer, model, device = _load_transformer(model_name)
    if gpu_mode is None:
        gpu_mode = gpu_xgboost_ready()
    embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding text")

    for index in iterator:
        batch = texts[index : index + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            if gpu_mode:
                embeddings.append(cp.asarray(cls_embeddings.detach()))
            else:
                embeddings.append(cls_embeddings.detach().cpu().numpy())

    if not embeddings:
        return cp.empty((0, 768)) if gpu_mode else np.empty((0, 768))
    return cp.concatenate(embeddings, axis=0) if gpu_mode else np.vstack(embeddings)


def split_embeddings(
    embeddings,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    if cp is not None and isinstance(embeddings, cp.ndarray):
        return (
            embeddings[cp.asarray(train_idx)],
            embeddings[cp.asarray(test_idx)],
            labels.iloc[train_idx].reset_index(drop=True),
            labels.iloc[test_idx].reset_index(drop=True),
        )
    return (
        embeddings[train_idx],
        embeddings[test_idx],
        labels.iloc[train_idx].reset_index(drop=True),
        labels.iloc[test_idx].reset_index(drop=True),
    )


def to_cpu_numpy(array_like):
    if cp is not None and isinstance(array_like, cp.ndarray):
        return cp.asnumpy(array_like)
    return array_like


def build_classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        device=get_xgboost_device(),
    )


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=5, zero_division=0, output_dict=True
        ),
    }


def cross_validate_classifier(
    embeddings: np.ndarray,
    labels: pd.Series,
    scale_pos_weight: float,
    folds: int = 5,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = {"accuracy": [], "balanced_accuracy": [], "f1": [], "roc_auc": []}

    for train_idx, test_idx in cv.split(np.arange(len(labels)), labels):
        classifier = build_classifier(scale_pos_weight=scale_pos_weight)
        if cp is not None and isinstance(embeddings, cp.ndarray):
            X_train = embeddings[cp.asarray(train_idx)]
            X_test = embeddings[cp.asarray(test_idx)]
        else:
            X_train = embeddings[train_idx]
            X_test = embeddings[test_idx]

        y_train = labels.iloc[train_idx].reset_index(drop=True)
        y_test = labels.iloc[test_idx].reset_index(drop=True)
        classifier.fit(X_train, y_train)

        y_pred = to_cpu_numpy(classifier.predict(X_test))
        y_prob = to_cpu_numpy(classifier.predict_proba(X_test)[:, 1])
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        results["roc_auc"].append(roc_auc_score(y_test, y_prob))

    return {
        metric: float(np.mean(values)) for metric, values in results.items()
    }


def plot_evaluation(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "roc_curve.png", dpi=150)
    plt.close()


def train_pipeline(
    csv_path: str | Path,
    artifact_dir: str | Path,
    reports_dir: str | Path,
    feature_method: str = "mutual_info",
    test_size: float = 0.2,
    embedding_model: str = "roberta-base",
) -> Dict[str, object]:
    df = load_dataset(csv_path)
    rankings = rank_feature_groups(df)
    if feature_method not in rankings:
        raise ValueError(
            f"Unknown feature selection method '{feature_method}'. "
            f"Choose from: {sorted(rankings)}"
        )

    selected_columns = rankings[feature_method]["best_columns"]
    prepared = build_model_inputs(df, selected_columns)
    texts = prepared["intent_text"].tolist()
    labels = prepared[TARGET_COLUMN]
    embeddings = get_embeddings_for_xgboost(texts, model_name=embedding_model)

    X_train, X_test, y_train, y_test = split_embeddings(
        embeddings, labels, test_size=test_size, random_state=42
    )
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    classifier = build_classifier(scale_pos_weight=ratio)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    y_pred_cpu = to_cpu_numpy(y_pred)
    y_prob_cpu = to_cpu_numpy(y_prob)
    holdout_metrics = evaluate_predictions(y_test, y_pred_cpu, y_prob_cpu)
    cv_metrics = cross_validate_classifier(embeddings, labels, scale_pos_weight=ratio)

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, artifact_path / "classifier.joblib")
    metadata = {
        "csv_path": str(csv_path),
        "feature_method": feature_method,
        "selected_columns": selected_columns,
        "embedding_model": embedding_model,
        "torch_device": str(get_torch_device()),
        "xgboost_device": get_xgboost_device(),
        "target_column": TARGET_COLUMN,
        "leakage_columns_removed": sorted(LEAKAGE_COLUMNS),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(prepared)),
        "cross_validation": cv_metrics,
        "holdout_metrics": holdout_metrics,
    }
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for name, payload in rankings.items():
        payload["table"].to_csv(reports_path / f"{name}_feature_ranking.csv", index=False)

    plot_evaluation(y_test, y_prob_cpu, y_pred_cpu, reports_path)
    return metadata


class DefectPredictor:
    def __init__(self, artifact_dir: str | Path):
        artifact_path = Path(artifact_dir)
        self.classifier = joblib.load(artifact_path / "classifier.joblib")
        self.metadata = json.loads((artifact_path / "metadata.json").read_text())
        self.selected_columns = self.metadata["selected_columns"]
        self.embedding_model = self.metadata["embedding_model"]
        self.inference_xgboost_device = (
            self.metadata.get("xgboost_device", "cpu")
            if gpu_xgboost_ready()
            else "cpu"
        )
        self.classifier.set_params(device=self.inference_xgboost_device)
        try:
            self.classifier.get_booster().set_param({"device": self.inference_xgboost_device})
        except Exception:
            pass

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        prepared = input_df.copy()
        for column in self.selected_columns:
            if column not in prepared.columns:
                prepared[column] = ""

        prepared["intent_text"] = build_intent_text(prepared, self.selected_columns)
        embeddings = get_embeddings_for_xgboost(
            prepared["intent_text"].tolist(),
            model_name=self.embedding_model,
            gpu_mode=self.inference_xgboost_device.startswith("cuda"),
            show_progress=False,
        )
        probabilities = self.classifier.predict_proba(embeddings)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        probabilities_cpu = to_cpu_numpy(probabilities)
        predictions_cpu = to_cpu_numpy(predictions)

        result = prepared[self.selected_columns + ["intent_text"]].copy()
        result["defect_probability"] = probabilities_cpu
        result["predicted_label"] = np.where(
            predictions_cpu == 1, "Defective", "Non-defective"
        )
        return result
