import os
import pandas as pd

from model_utils import (
    DEFAULT_MODEL_PATHS,
    WINNER_BASE_FEATURES,
    WINNER_ADVANCED_FEATURES,
    load_clean_data,
    load_runtime_artifacts,
)


def build_latest_stats(df, fighter_name):
    fights = df[(df["R_fighter"] == fighter_name) | (df["B_fighter"] == fighter_name)].sort_values("date")
    if fights.empty:
        return None

    row = fights.iloc[-1]
    prefix = "R_" if row["R_fighter"] == fighter_name else "B_"
    return {
        "age": row.get(f"{prefix}age"),
        "wins": row.get(f"{prefix}wins"),
        "streak": row.get(f"{prefix}current_win_streak"),
        "form": row.get(f"{prefix}recent_form"),
        "height": row.get(f"{prefix}Height_cms"),
        "reach": row.get(f"{prefix}Reach_cms"),
        "avg_SIG_STR_landed": row.get(f"{prefix}avg_SIG_STR_landed"),
        "avg_TD_landed": row.get(f"{prefix}avg_TD_landed"),
        "avg_SUB_ATT": row.get(f"{prefix}avg_SUB_ATT"),
        "avg_KD": row.get(f"{prefix}avg_KD"),
        "avg_CTRL_time": row.get(f"{prefix}avg_CTRL_time(seconds)"),
        "avg_opp_SIG_STR_pct": row.get(f"{prefix}avg_opp_SIG_STR_pct"),
        "avg_opp_TD_pct": row.get(f"{prefix}avg_opp_TD_pct"),
        "total_rounds_fought": row.get(f"{prefix}total_rounds_fought"),
        "total_time_fought": row.get(f"{prefix}total_time_fought(seconds)"),
        "days_since_last_fight": row.get(f"{prefix}days_since_last_fight"),
    }


def build_diff_frame(red_stats, blue_stats):
    diffs = {
        "diff_recent_form": red_stats["form"] - blue_stats["form"],
        "diff_wins": red_stats["wins"] - blue_stats["wins"],
        "diff_current_win_streak": red_stats["streak"] - blue_stats["streak"],
        "diff_age": red_stats["age"] - blue_stats["age"],
        "diff_height": red_stats["height"] - blue_stats["height"],
        "diff_reach": red_stats["reach"] - blue_stats["reach"],
        "diff_avg_SIG_STR_landed": red_stats["avg_SIG_STR_landed"] - blue_stats["avg_SIG_STR_landed"],
        "diff_avg_TD_landed": red_stats["avg_TD_landed"] - blue_stats["avg_TD_landed"],
        "diff_avg_SUB_ATT": red_stats["avg_SUB_ATT"] - blue_stats["avg_SUB_ATT"],
        "diff_avg_KD": red_stats["avg_KD"] - blue_stats["avg_KD"],
        "diff_avg_CTRL_time": red_stats["avg_CTRL_time"] - blue_stats["avg_CTRL_time"],
        "diff_sig_str_defense_proxy": blue_stats["avg_opp_SIG_STR_pct"] - red_stats["avg_opp_SIG_STR_pct"],
        "diff_td_defense_proxy": blue_stats["avg_opp_TD_pct"] - red_stats["avg_opp_TD_pct"],
        "diff_total_rounds_fought": red_stats["total_rounds_fought"] - blue_stats["total_rounds_fought"],
        "diff_total_time_fought": red_stats["total_time_fought"] - blue_stats["total_time_fought"],
        "diff_days_since_last_fight": red_stats["days_since_last_fight"] - blue_stats["days_since_last_fight"],
    }
    return pd.DataFrame([diffs])


def ensure_artifacts():
    base_dir = os.path.dirname(__file__)
    requested_paths = {
        key: os.path.join(base_dir, "..", "data", os.path.basename(path))
        for key, path in DEFAULT_MODEL_PATHS.items()
    }
    model_paths, _, _, _, _, _ = load_runtime_artifacts(output_paths=requested_paths)
    return model_paths


def show_summary(summary):
    print("\n" + "=" * 60)
    print("MODELLSUMMARY")
    print("=" * 60)
    print("Datensatz:", summary.get("dataset"))
    print("Winner Baseline:", summary.get("winner_baseline"))
    print("Winner Advanced:", summary.get("winner_advanced"))
    print("Empfohlen:", summary.get("recommended_winner_model"))
    if summary.get("round_model"):
        print("Round Model:", summary["round_model"])
    if summary.get("method_model"):
        print("Method Model:", summary["method_model"])


def predict_head_to_head(df, baseline_artifact, advanced_artifact):
    red_name = input("Rote Ecke: ").strip()
    blue_name = input("Blaue Ecke: ").strip()
    red_stats = build_latest_stats(df, red_name)
    blue_stats = build_latest_stats(df, blue_name)

    if not red_stats or not blue_stats:
        print("Mindestens ein Fighter konnte nicht gefunden werden.")
        return

    X = build_diff_frame(red_stats, blue_stats)
    predictions = [
        ("Baseline", baseline_artifact, WINNER_BASE_FEATURES),
        ("Advanced", advanced_artifact, WINNER_ADVANCED_FEATURES),
    ]

    for label, artifact, features in predictions:
        model = artifact["model"]
        result = model.predict(X[features])[0]
        proba = model.predict_proba(X[features])[0]
        winner = red_name if result == 1 else blue_name
        print(f"\n{label}: {winner}")
        print(f"  {red_name}: {proba[1]*100:.1f}%")
        print(f"  {blue_name}: {proba[0]*100:.1f}%")


def test_model():
    df = load_clean_data()
    _, baseline_artifact, advanced_artifact, _, _, summary = load_runtime_artifacts()
    show_summary(summary)

    while True:
        print("\n[1] Modellsummary anzeigen")
        print("[2] Zwei Fighter vergleichen")
        print("[3] Beenden")
        choice = input("Auswahl: ").strip()

        if choice == "1":
            show_summary(summary)
        elif choice == "2":
            predict_head_to_head(df, baseline_artifact, advanced_artifact)
        elif choice == "3":
            break
        else:
            print("Ungültige Auswahl.")


if __name__ == "__main__":
    test_model()
