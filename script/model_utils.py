import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "cleaned_ufc_data.csv")
DATASET_METADATA_PATH = os.path.join(DATA_DIR, "dataset_metadata.json")
MODEL_SUMMARY_PATH = os.path.join(DATA_DIR, "model_summary.json")

WINNER_BASE_FEATURES = [
    "diff_recent_form",
    "diff_wins",
    "diff_current_win_streak",
    "diff_age",
    "diff_days_since_last_fight",
]

WINNER_ADVANCED_FEATURES = [
    "diff_recent_form",
    "diff_wins",
    "diff_current_win_streak",
    "diff_age",
    "diff_height",
    "diff_reach",
    "diff_avg_SIG_STR_landed",
    "diff_avg_TD_landed",
    "diff_avg_SUB_ATT",
    "diff_avg_KD",
    "diff_avg_CTRL_time",
    "diff_sig_str_defense_proxy",
    "diff_td_defense_proxy",
    "diff_total_rounds_fought",
    "diff_total_time_fought",
    "diff_days_since_last_fight",
]

ROUND_FEATURES = [
    "diff_recent_form",
    "diff_current_win_streak",
    "diff_avg_SIG_STR_landed",
    "diff_avg_TD_landed",
    "diff_avg_SUB_ATT",
    "diff_avg_KD",
    "diff_avg_CTRL_time",
    "diff_sig_str_defense_proxy",
    "diff_td_defense_proxy",
    "diff_total_rounds_fought",
]

METHOD_FEATURES = ROUND_FEATURES

DEFAULT_MODEL_PATHS = {
    "winner_baseline": os.path.join(DATA_DIR, "baseline_model.pkl"),
    "winner_advanced": os.path.join(DATA_DIR, "advanced_model.pkl"),
    "fight_round": os.path.join(DATA_DIR, "round_model.pkl"),
    "fight_method": os.path.join(DATA_DIR, "method_model.pkl"),
}


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_clean_data(data_path=CLEAN_DATA_PATH):
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def ensure_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_source_version(df):
    max_date = None
    if "date" in df.columns:
        max_date = pd.to_datetime(df["date"], errors="coerce").max()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "source_max_event_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
    }


def time_split_frame(df, min_test_size=200, test_fraction=0.2):
    ordered = df.sort_values("date").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("Zu wenige Datenpunkte für einen Train/Test-Split.")

    test_size = max(min_test_size, int(len(ordered) * test_fraction))
    test_size = min(test_size, len(ordered) - 1)
    split_idx = len(ordered) - test_size
    return TemporalSplit(train=ordered.iloc[:split_idx].copy(), test=ordered.iloc[split_idx:].copy())


def build_logistic_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def build_random_forest_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def simplify_finish_method(method_name):
    if not isinstance(method_name, str):
        return np.nan
    lowered = method_name.lower()
    if "decision" in lowered:
        return "Decision"
    if "sub" in lowered:
        return "Submission"
    if "ko" in lowered or "tko" in lowered:
        return "KO / TKO"
    return "Other"


def evaluate_binary_artifact(model, X_test, y_test):
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    return metrics


def evaluate_multiclass_artifact(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def build_artifact(task_name, model_name, model, features, split, metrics, source_version):
    return {
        "task": task_name,
        "model_name": model_name,
        "model": model,
        "features": list(features),
        "metrics": metrics,
        "train_end_date": split.train["date"].max().strftime("%Y-%m-%d"),
        "test_start_date": split.test["date"].min().strftime("%Y-%m-%d"),
        "source_version": source_version,
    }


def train_winner_artifacts(df):
    source_version = get_source_version(df)
    artifacts = {}

    specs = [
        ("winner_baseline", "Baseline (Logistic Regression)", WINNER_BASE_FEATURES, build_logistic_pipeline()),
        ("winner_advanced", "Advanced (Random Forest)", WINNER_ADVANCED_FEATURES, build_random_forest_pipeline()),
    ]

    for key, model_name, features, model in specs:
        numeric_cols = list(features) + ["target"]
        data = df.dropna(subset=["target", "date"]).copy()
        data = ensure_numeric_columns(data, numeric_cols)
        data = data.dropna(subset=["target"])
        split = time_split_frame(data[["date", "target"] + features].copy())

        X_train = split.train[features]
        y_train = split.train["target"].astype(int)
        X_test = split.test[features]
        y_test = split.test["target"].astype(int)

        model.fit(X_train, y_train)
        metrics = evaluate_binary_artifact(model, X_test, y_test)

        baseline_dummy = DummyClassifier(strategy="most_frequent")
        baseline_dummy.fit(X_train, y_train)
        dummy_metrics = evaluate_binary_artifact(baseline_dummy, X_test, y_test)
        metrics["majority_class_accuracy"] = dummy_metrics["accuracy"]
        metrics["majority_class_balanced_accuracy"] = dummy_metrics["balanced_accuracy"]
        metrics["train_rows"] = int(len(split.train))
        metrics["test_rows"] = int(len(split.test))

        artifacts[key] = build_artifact(key, model_name, model, features, split, metrics, source_version)

    return artifacts


def train_round_artifact(df):
    if "finish_round" not in df.columns:
        return None

    source_version = get_source_version(df)
    data = df.dropna(subset=["date", "finish_round"]).copy()
    data = ensure_numeric_columns(data, ROUND_FEATURES + ["finish_round"])
    data["finish_round"] = data["finish_round"].astype("Int64")
    data = data[data["finish_round"].between(1, 5, inclusive="both")]

    if len(data) < 300:
        return None

    split = time_split_frame(data[["date", "finish_round"] + ROUND_FEATURES].copy())
    model = build_random_forest_pipeline()
    X_train = split.train[ROUND_FEATURES]
    y_train = split.train["finish_round"].astype(int)
    X_test = split.test[ROUND_FEATURES]
    y_test = split.test["finish_round"].astype(int)

    model.fit(X_train, y_train)
    metrics = evaluate_multiclass_artifact(model, X_test, y_test)
    metrics["train_rows"] = int(len(split.train))
    metrics["test_rows"] = int(len(split.test))
    return build_artifact("fight_round", "Fight Round (Random Forest)", model, ROUND_FEATURES, split, metrics, source_version)


def train_method_artifact(df):
    if "finish_method" not in df.columns:
        return None

    source_version = get_source_version(df)
    data = df.dropna(subset=["date", "finish_method"]).copy()
    data["finish_method_simple"] = data["finish_method"].apply(simplify_finish_method)
    data = data[data["finish_method_simple"].isin(["Decision", "Submission", "KO / TKO"])]

    if len(data) < 300:
        return None

    split = time_split_frame(data[["date", "finish_method_simple"] + METHOD_FEATURES].copy())
    model = build_random_forest_pipeline()
    X_train = split.train[METHOD_FEATURES]
    y_train = split.train["finish_method_simple"]
    X_test = split.test[METHOD_FEATURES]
    y_test = split.test["finish_method_simple"]

    model.fit(X_train, y_train)
    metrics = {
        "accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, model.predict(X_test))),
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
    }
    return build_artifact("fight_method", "Fight Method (Random Forest)", model, METHOD_FEATURES, split, metrics, source_version)


def save_artifact(path, artifact):
    with open(path, "wb") as handle:
        pickle.dump(artifact, handle)


def load_artifact(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_model_summary(summary, path=MODEL_SUMMARY_PATH):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)


def load_model_summary(path=MODEL_SUMMARY_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_feature_importances(artifact):
    model = artifact["model"]
    if not hasattr(model, "named_steps"):
        return {}
    final_model = model.named_steps.get("model")
    if not hasattr(final_model, "feature_importances_"):
        return {}
    return {
        name: float(score)
        for name, score in sorted(
            zip(artifact["features"], final_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def train_all_models(data_path=CLEAN_DATA_PATH, output_paths=None):
    df = load_clean_data(data_path)
    output_paths = output_paths or DEFAULT_MODEL_PATHS

    winner_artifacts = train_winner_artifacts(df)
    round_artifact = train_round_artifact(df)
    method_artifact = train_method_artifact(df)

    for key, artifact in winner_artifacts.items():
        save_artifact(output_paths[key], artifact)
    if round_artifact:
        save_artifact(output_paths["fight_round"], round_artifact)
    if method_artifact:
        save_artifact(output_paths["fight_method"], method_artifact)

    advanced_artifact = winner_artifacts["winner_advanced"]
    summary = {
        "dataset": get_source_version(df),
        "winner_baseline": winner_artifacts["winner_baseline"]["metrics"],
        "winner_advanced": winner_artifacts["winner_advanced"]["metrics"],
        "recommended_winner_model": (
            "winner_advanced"
            if winner_artifacts["winner_advanced"]["metrics"]["balanced_accuracy"]
            >= winner_artifacts["winner_baseline"]["metrics"]["balanced_accuracy"]
            else "winner_baseline"
        ),
        "round_model": round_artifact["metrics"] if round_artifact else None,
        "method_model": method_artifact["metrics"] if method_artifact else None,
        "advanced_feature_importances": extract_feature_importances(advanced_artifact),
    }
    save_model_summary(summary)

    return {
        "winner_baseline": winner_artifacts["winner_baseline"],
        "winner_advanced": winner_artifacts["winner_advanced"],
        "fight_round": round_artifact,
        "fight_method": method_artifact,
        "summary": summary,
    }
