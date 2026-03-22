import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import time
import warnings
from datetime import date
warnings.filterwarnings('ignore')

from model_utils import (
    DEFAULT_MODEL_PATHS,
    METHOD_FEATURES,
    ROUND_FEATURES,
    WINNER_ADVANCED_FEATURES,
    WINNER_BASE_FEATURES,
    load_artifact,
    load_clean_data,
    load_model_summary,
    load_runtime_artifacts,
    train_all_models,
)

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.fighter-card {
    border-radius: 16px;
    padding: 24px 20px 16px 20px;
    text-align: center;
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    margin-bottom: 12px;
    border: 2px solid #333;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.red-card  { border-color: #ff4b4b; box-shadow: 0 8px 32px rgba(255,75,75,0.25); }
.blue-card { border-color: #4b8bff; box-shadow: 0 8px 32px rgba(75,139,255,0.25); }

.fighter-name { font-size: 22px; font-weight: 900; margin: 10px 0 4px 0; }
.red-card  .fighter-name { color: #ff4b4b; }
.blue-card .fighter-name { color: #4b8bff; }
.stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2a2a3e; font-size: 13px; }
.stat-label { color: #888; }
.stat-val   { color: #eee; font-weight: 600; }
.stat-streak { color: #ffd700; font-weight: 700; }

.verdict-box {
    border-radius: 16px; padding: 24px;
    text-align: center; margin: 16px 0;
    background: linear-gradient(135deg, #1a1a2e, #0d0d1a);
    border: 2px solid #555;
}
.verdict-red  { border-color: #ff4b4b; }
.verdict-blue { border-color: #4b8bff; }
.verdict-name { font-size: 28px; font-weight: 900; margin-bottom: 6px; }

.recommend-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;
    border-left: 5px solid #00e5ff;
    color: #fff;
}
.why-box {
    background: #111827; border-radius: 12px; padding: 16px 20px;
    border-left: 5px solid #7c3aed; margin-top: 12px; color: #ddd;
    font-size: 14px; line-height: 1.7;
}
.feat-row { display: flex; align-items: center; margin: 5px 0; font-size: 13px; }
.feat-label { width: 220px; color: #aaa; }
.feat-bar-wrap { flex: 1; height: 12px; background: #222; border-radius: 6px; overflow: hidden; margin: 0 8px; }
.feat-bar-pos { height: 100%; background: #ff4b4b; border-radius: 6px; }
.feat-bar-neg { height: 100%; background: #4b8bff; border-radius: 6px; margin-left: auto; }
.feat-val { width: 60px; text-align: right; color: #ccc; font-weight: 600; }

.matchup-shell {
    margin: 10px 0 18px 0;
    padding: 18px;
    border-radius: 22px;
    background: linear-gradient(160deg, #101625, #121a2f 52%, #0f1424);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
}
.fighter-panel {
    border-radius: 18px;
    padding: 18px;
    min-height: 150px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
}
.red-panel { box-shadow: inset 0 0 0 1px rgba(255,75,75,0.18); }
.blue-panel { box-shadow: inset 0 0 0 1px rgba(75,139,255,0.18); }
.corner-chip {
    display: inline-block;
    font-size: 11px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #8d99ae;
    margin-bottom: 6px;
}
.panel-name {
    font-size: 28px;
    font-weight: 900;
    line-height: 1.05;
    margin-bottom: 6px;
}
.red-panel .panel-name { color: #ff6b6b; }
.blue-panel .panel-name { color: #6ea8ff; }
.panel-record {
    color: #c8d1e8;
    font-size: 13px;
    margin-bottom: 10px;
}
.panel-finish {
    margin-bottom: 14px;
    color: #f4d58d;
    font-size: 12px;
    line-height: 1.5;
}
.panel-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.panel-badge {
    padding: 7px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    color: #dbe4ff;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
.panel-badge-prime {
    background: linear-gradient(135deg, rgba(255,215,0,0.22), rgba(255,184,28,0.4));
    color: #fff2b3;
    border-color: rgba(255,215,0,0.55);
    box-shadow: 0 0 18px rgba(255,215,0,0.18);
    font-weight: 700;
}
.prime-note {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #d7dfef;
    font-size: 12px;
    line-height: 1.5;
}
.prime-note strong {
    color: #ffd86b;
}
.compare-card {
    margin-top: 14px;
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.06);
}
.compare-title {
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #97a3bf;
    margin-bottom: 12px;
}
.compare-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}
.compare-item {
    border-radius: 14px;
    padding: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.compare-label {
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #8d99ae;
    margin-bottom: 8px;
}
.compare-main {
    font-size: 20px;
    font-weight: 800;
    color: #edf2ff;
    margin-bottom: 6px;
}
.compare-sub {
    color: #c8d1e8;
    font-size: 13px;
    line-height: 1.5;
}
.compare-note {
    margin-top: 12px;
    color: #c8d1e8;
    font-size: 12px;
    line-height: 1.6;
}
.vs-core {
    height: 100%;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background: radial-gradient(circle at top, rgba(255,255,255,0.08), transparent 55%), linear-gradient(180deg, #17142c, #111827);
    border: 1px solid rgba(255,255,255,0.08);
}
.vs-badge {
    width: 84px;
    height: 84px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: 900;
    color: #fff;
    background: linear-gradient(135deg, #ff4b4b, #5f8fff);
    box-shadow: 0 10px 30px rgba(79,104,255,0.30);
    margin-bottom: 12px;
}
.vs-caption {
    font-size: 12px;
    letter-spacing: 2px;
    color: #97a3bf;
    text-transform: uppercase;
}
.tape-card {
    margin-top: 14px;
    padding: 18px 20px 10px 20px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.06);
}
.tape-title {
    text-align: center;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #97a3bf;
    margin-bottom: 12px;
}
.tape-row {
    display: grid;
    grid-template-columns: 88px 1fr 130px 1fr 88px;
    gap: 12px;
    align-items: center;
    padding: 10px 0;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.tape-row:first-of-type { border-top: none; }
.tape-value-left, .tape-value-right {
    font-weight: 700;
    color: #edf2ff;
    font-size: 14px;
}
.tape-value-left { text-align: right; }
.tape-value-right { text-align: left; }
.tape-adv-left { color: #ff8585; }
.tape-adv-right { color: #7fb0ff; }
.tape-label {
    text-align: center;
    color: #9aa6c4;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.tape-bar {
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}
.tape-fill-red, .tape-fill-blue {
    position: absolute;
    top: 0;
    height: 100%;
}
.tape-fill-red {
    right: 0;
    background: linear-gradient(90deg, rgba(255,75,75,0.25), #ff4b4b);
}
.tape-fill-blue {
    left: 0;
    background: linear-gradient(90deg, #4b8bff, rgba(75,139,255,0.25));
}
.tape-label-adv-left { color: #ff8585; font-weight: 700; }
.tape-label-adv-right { color: #7fb0ff; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FIGHTER PHOTOS – lokale Bilder aus JSON-Mapping laden
# ─────────────────────────────────────────────────────────────
_PHOTOS_JSON = os.path.join(
    os.path.dirname(__file__), "..", "data", "fighter_images", "fighter_photos.json"
)
try:
    with open(_PHOTOS_JSON, encoding="utf-8") as _f:
        FIGHTER_PHOTOS: dict[str, str] = json.load(_f)
except Exception:
    FIGHTER_PHOTOS = {}

PLACEHOLDER_IMG = os.path.join(
    os.path.dirname(__file__), "..", "data", "fighter_images", "placeholder.png"
)

FIGHTER_DETAILS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "row_fighter_details.csv"
)
BOOKMAKER_ODDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "bookmaker_odds.csv"
)

# ─────────────────────────────────────────────────────────────
# MODELL-FEATURES
# ─────────────────────────────────────────────────────────────
BASE_FEATURES = WINNER_BASE_FEATURES
ADV_FEATURES = WINNER_ADVANCED_FEATURES
TARGET = 'target'
ROUND_TARGET  = 'finish_round'
METHOD_TARGET = 'finish_method'

FEATURE_LABELS = {
    'diff_recent_form':         'Form-Differenz (letzte 3 Kämpfe)',
    'diff_wins':                'Siege-Differenz',
    'diff_current_win_streak':  'Aktuelle Siegesserie',
    'diff_age':                 'Altersunterschied',
    'diff_height':              'Größenunterschied',
    'diff_reach':               'Reichweitenunterschied',
    'diff_avg_SIG_STR_landed':  'Ø Significant Strikes',
    'diff_avg_TD_landed':       'Ø Takedowns',
    'diff_avg_SUB_ATT':         'Ø Submission-Versuche',
    'diff_avg_KD':              'Ø Knockdowns',
    'diff_avg_CTRL_time':       'Ø Bodenkontrolle (s)',
    'diff_sig_str_defense_proxy': 'Striking-Defense-Proxy',
    'diff_td_defense_proxy':    'Takedown-Defense-Proxy',
    'diff_total_rounds_fought': 'Erfahrung in Runden',
    'diff_total_time_fought':   'Gesamtkampfzeit',
    'diff_days_since_last_fight': 'Inaktivität bis zum Stichtag',
}

# ─────────────────────────────────────────────────────────────
# DATEN & MODELLE LADEN / TRAINIEREN
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_everything():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "cleaned_ufc_data.csv")
    df = load_clean_data(data_path)
    model_paths = {
        key: os.path.join(base_dir, "..", "data", os.path.basename(path))
        for key, path in DEFAULT_MODEL_PATHS.items()
    }
    _, baseline_artifact, advanced_artifact, round_artifact, method_artifact, summary = load_runtime_artifacts(
        data_path=data_path,
        output_paths=model_paths,
    )

    return df, baseline_artifact, advanced_artifact, round_artifact, method_artifact, summary


@st.cache_data
def get_fighters(df):
    fighters = set(df['R_fighter'].dropna()).union(set(df['B_fighter'].dropna()))
    return sorted(fighters)


@st.cache_data
def load_fighter_bios():
    if not os.path.exists(FIGHTER_DETAILS_PATH):
        return {}

    bios = pd.read_csv(FIGHTER_DETAILS_PATH, usecols=["fighter_name", "DOB"])
    bios["DOB_dt"] = pd.to_datetime(bios["DOB"], errors="coerce")
    bios = bios.dropna(subset=["fighter_name"]).drop_duplicates(subset=["fighter_name"], keep="first")
    return {
        row["fighter_name"]: row["DOB_dt"]
        for _, row in bios.iterrows()
        if pd.notna(row["DOB_dt"])
    }


def compute_current_age(name, reference_date=None, fallback_age=np.nan):
    dob = load_fighter_bios().get(name)
    if pd.notna(dob):
        age_reference = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp(date.today())
        return round((age_reference - dob).days / 365.25, 2)
    return fallback_age


def normalize_fighter_name(name):
    cleaned = str(name or "").strip().lower()
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace("-", " ")
    return " ".join(cleaned.split())


@st.cache_data
def _load_bookmaker_odds_cached(_cache_token):
    columns = [
        "event_name", "event_date", "bookmaker_name", "market",
        "fighter_a", "fighter_b", "fighter_a_decimal_odds", "fighter_b_decimal_odds",
        "fighter_a_implied_prob", "fighter_b_implied_prob", "bookmaker_reason", "odds_timestamp",
    ]
    if not os.path.exists(BOOKMAKER_ODDS_PATH):
        return pd.DataFrame(columns=columns)

    odds = pd.read_csv(BOOKMAKER_ODDS_PATH)
    for column in columns:
        if column not in odds.columns:
            odds[column] = np.nan
    odds["event_date"] = pd.to_datetime(odds["event_date"], errors="coerce")
    odds["odds_timestamp"] = pd.to_datetime(odds["odds_timestamp"], errors="coerce")
    odds["fighter_a_norm"] = odds["fighter_a"].map(normalize_fighter_name)
    odds["fighter_b_norm"] = odds["fighter_b"].map(normalize_fighter_name)
    return odds


def load_bookmaker_odds():
    cache_token = None
    if os.path.exists(BOOKMAKER_ODDS_PATH):
        stat = os.stat(BOOKMAKER_ODDS_PATH)
        cache_token = (stat.st_mtime_ns, stat.st_size)
    return _load_bookmaker_odds_cached(cache_token)


def find_actual_fight(df, red_name, blue_name, reference_date):
    fights = df[
        (
            ((df["R_fighter"] == red_name) & (df["B_fighter"] == blue_name))
            | ((df["R_fighter"] == blue_name) & (df["B_fighter"] == red_name))
        )
    ].copy()
    if fights.empty:
        return None

    fights["date"] = pd.to_datetime(fights["date"], errors="coerce")
    ref_ts = pd.Timestamp(reference_date) if reference_date is not None else fights["date"].max()
    fights = fights[fights["date"] <= ref_ts].sort_values("date", ascending=False)
    if fights.empty:
        return None
    return fights.iloc[0]


def format_decimal_odds(value):
    if pd.isna(value):
        return "k.A."
    return f"{float(value):.2f}"


def format_pct(value):
    if pd.isna(value):
        return "k.A."
    return f"{float(value) * 100:.1f}%"


def lookup_bookmaker_context(df, red_name, blue_name, reference_date):
    odds = load_bookmaker_odds()
    actual_fight = find_actual_fight(df, red_name, blue_name, reference_date)
    if odds.empty:
        return {"odds": None, "actual_fight": actual_fight}

    red_norm = normalize_fighter_name(red_name)
    blue_norm = normalize_fighter_name(blue_name)
    matches = odds[
        (
            ((odds["fighter_a_norm"] == red_norm) & (odds["fighter_b_norm"] == blue_norm))
            | ((odds["fighter_a_norm"] == blue_norm) & (odds["fighter_b_norm"] == red_norm))
        )
    ].copy()
    if matches.empty:
        return {"odds": None, "actual_fight": actual_fight}

    if actual_fight is not None and pd.notna(actual_fight["date"]):
        exact_date_matches = matches[matches["event_date"] == pd.Timestamp(actual_fight["date"])]
        if not exact_date_matches.empty:
            matches = exact_date_matches

    ref_ts = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.max
    valid_by_date = matches[matches["event_date"].isna() | (matches["event_date"] <= ref_ts)]
    if not valid_by_date.empty:
        matches = valid_by_date

    matches["completeness_score"] = (
        matches["fighter_a_implied_prob"].notna().astype(int)
        + matches["fighter_b_implied_prob"].notna().astype(int)
    )
    matches = matches.sort_values(
        ["completeness_score", "event_date", "odds_timestamp"],
        ascending=[False, False, False],
        na_position="last",
    )
    best = matches.iloc[0]
    swapped = best["fighter_a_norm"] != red_norm

    red_odds = best["fighter_b_decimal_odds"] if swapped else best["fighter_a_decimal_odds"]
    blue_odds = best["fighter_a_decimal_odds"] if swapped else best["fighter_b_decimal_odds"]
    red_prob = best["fighter_b_implied_prob"] if swapped else best["fighter_a_implied_prob"]
    blue_prob = best["fighter_a_implied_prob"] if swapped else best["fighter_b_implied_prob"]
    if pd.notna(red_prob) and pd.notna(blue_prob):
        favorite = red_name if float(red_prob) >= float(blue_prob) else blue_name
    elif pd.notna(red_prob):
        favorite = red_name
    elif pd.notna(blue_prob):
        favorite = blue_name
    elif pd.notna(red_odds) and pd.notna(blue_odds):
        favorite = red_name if float(red_odds) <= float(blue_odds) else blue_name
    else:
        favorite = "k.A."

    return {
        "odds": {
            "event_name": best.get("event_name"),
            "event_date": best.get("event_date"),
            "bookmaker_name": best.get("bookmaker_name"),
            "market": best.get("market"),
            "red_odds": red_odds,
            "blue_odds": blue_odds,
            "red_prob": red_prob,
            "blue_prob": blue_prob,
            "favorite": favorite,
            "reason": best.get("bookmaker_reason"),
        },
        "actual_fight": actual_fight,
    }


def render_prediction_comparison(model_pick, model_prob, bookmaker_context, actual_fight, red_name, blue_name):
    bookmaker = bookmaker_context.get("odds") if bookmaker_context else None
    if actual_fight is not None and pd.notna(actual_fight.get("date")):
        if actual_fight["R_fighter"] == red_name:
            actual_winner = red_name if actual_fight["Winner"] == "Red" else blue_name if actual_fight["Winner"] == "Blue" else "Draw / NC"
        else:
            actual_winner = blue_name if actual_fight["Winner"] == "Red" else red_name if actual_fight["Winner"] == "Blue" else "Draw / NC"
        actual_main = actual_winner
        actual_sub = pd.Timestamp(actual_fight["date"]).strftime("%Y-%m-%d")
    else:
        actual_main = "Offen"
        actual_sub = "Noch kein Ergebnis für diesen Fight"

    if bookmaker:
        bookmaker_main = bookmaker["favorite"]
        bookmaker_sub = (
            f"{red_name}: {format_decimal_odds(bookmaker['red_odds'])} ({format_pct(bookmaker['red_prob'])})<br>"
            f"{blue_name}: {format_decimal_odds(bookmaker['blue_odds'])} ({format_pct(bookmaker['blue_prob'])})<br>"
            f"{bookmaker.get('bookmaker_name') or 'Bookmaker'}"
        )
        bookmaker_note = bookmaker.get("reason") or "Keine externe Buchmacher-Erklärung verfügbar."
    else:
        bookmaker_main = "Keine echten Odds"
        bookmaker_sub = "Für dieses Matchup wurde kein sauber zuordenbarer Buchmacher-Markt gefunden."
        bookmaker_note = "Buchmacher-Daten werden nur bei realen, eindeutig gematchten Kämpfen angezeigt."

    st.markdown(
        f"""
        <div class="compare-card">
            <div class="compare-title">Modell vs. Buchmacher vs. Outcome</div>
            <div class="compare-grid">
                <div class="compare-item">
                    <div class="compare-label">Unser Modell</div>
                    <div class="compare-main">{model_pick}</div>
                    <div class="compare-sub">{format_pct(model_prob)} Siegwahrscheinlichkeit</div>
                </div>
                <div class="compare-item">
                    <div class="compare-label">Buchmacher</div>
                    <div class="compare-main">{bookmaker_main}</div>
                    <div class="compare-sub">{bookmaker_sub}</div>
                </div>
                <div class="compare-item">
                    <div class="compare-label">Tatsächliches Ergebnis</div>
                    <div class="compare-main">{actual_main}</div>
                    <div class="compare-sub">{actual_sub}</div>
                </div>
            </div>
            <div class="compare-note"><strong>Bookmaker-Notiz:</strong> {bookmaker_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_stats(df, name, reference_date=None):
    fights = df[(df['R_fighter'] == name) | (df['B_fighter'] == name)].copy()
    fights['date'] = pd.to_datetime(fights['date'])
    if reference_date is not None:
        ref_ts = pd.Timestamp(reference_date)
        fights = fights[fights['date'] <= ref_ts]
    fights = fights.sort_values('date')
    if fights.empty:
        return None
    row = fights.iloc[-1]
    side = 'R_' if row['R_fighter'] == name else 'B_'
    p = side

    # Die CSV enthält Pre-Fight-Stats (VOR dem letzten Kampf).
    # Wir korrigieren Wins/Losses/Streak auf den Post-Fight-Stand.
    winner = row.get('Winner')
    won_last = (winner == 'Red' and side == 'R_') or (winner == 'Blue' and side == 'B_')

    pre_wins   = row.get(f'{p}wins', np.nan)
    pre_losses = row.get(f'{p}losses', np.nan)
    pre_streak = row.get(f'{p}current_win_streak', np.nan)
    pre_longest = row.get(f'{p}longest_win_streak', np.nan)
    pre_ko_wins = row.get(f'{p}win_by_KO/TKO', np.nan)
    pre_sub_wins = row.get(f'{p}win_by_Submission', np.nan)
    pre_doc_wins = row.get(f'{p}win_by_TKO_Doctor_Stoppage', np.nan)
    finish_method = str(row.get('finish_method', '')).strip()

    if won_last:
        wins   = pre_wins + 1   if pd.notna(pre_wins)   else np.nan
        losses = pre_losses      if pd.notna(pre_losses) else np.nan
        streak = pre_streak + 1  if pd.notna(pre_streak) else np.nan
        longest = max(pre_longest, streak) if pd.notna(pre_longest) and pd.notna(streak) else pre_longest
        ko_wins = pre_ko_wins + 1 if pd.notna(pre_ko_wins) and finish_method == 'KO/TKO' else pre_ko_wins
        sub_wins = pre_sub_wins + 1 if pd.notna(pre_sub_wins) and finish_method == 'Submission' else pre_sub_wins
        doc_wins = pre_doc_wins + 1 if pd.notna(pre_doc_wins) and finish_method == 'TKO - Doctor\'s Stoppage' else pre_doc_wins
    else:
        wins   = pre_wins        if pd.notna(pre_wins)   else np.nan
        losses = pre_losses + 1  if pd.notna(pre_losses) else np.nan
        streak = 0
        longest = pre_longest
        ko_wins = pre_ko_wins
        sub_wins = pre_sub_wins
        doc_wins = pre_doc_wins

    total_finishes = (
        (ko_wins if pd.notna(ko_wins) else 0)
        + (sub_wins if pd.notna(sub_wins) else 0)
        + (doc_wins if pd.notna(doc_wins) else 0)
    )
    finish_rate = (total_finishes / wins) if pd.notna(wins) and wins > 0 else np.nan
    if reference_date is not None:
        days_since_last_fight = max((pd.Timestamp(reference_date) - row['date']).days, 0)
    else:
        days_since_last_fight = row.get(f'{p}days_since_last_fight', np.nan)

    current_age = compute_current_age(
        name,
        reference_date=reference_date,
        fallback_age=row.get(f'{p}age', np.nan),
    )

    return {
        'age':                 current_age,
        'wins':                wins,
        'losses':              losses,
        'streak':              streak,
        'longest_streak':      longest,
        'form':                row.get(f'{p}recent_form', np.nan),
        'height':              row.get(f'{p}Height_cms', np.nan),
        'reach':               row.get(f'{p}Reach_cms', np.nan),
        'avg_SIG_STR_landed':  row.get(f'{p}avg_SIG_STR_landed', np.nan),
        'avg_TD_landed':       row.get(f'{p}avg_TD_landed', np.nan),
        'avg_SUB_ATT':         row.get(f'{p}avg_SUB_ATT', np.nan),
        'avg_KD':              row.get(f'{p}avg_KD', np.nan),
        'avg_CTRL_time':       row.get(f'{p}avg_CTRL_time(seconds)', np.nan),
        'avg_opp_SIG_STR_pct': row.get(f'{p}avg_opp_SIG_STR_pct', np.nan),
        'avg_opp_TD_pct':      row.get(f'{p}avg_opp_TD_pct', np.nan),
        'total_rounds_fought': row.get(f'{p}total_rounds_fought', np.nan),
        'total_time_fought':   row.get(f'{p}total_time_fought(seconds)', np.nan),
        'days_since_last_fight': days_since_last_fight,
        'ko_wins':             ko_wins,
        'sub_wins':            sub_wins,
        'finish_rate':         finish_rate,
    }


def safe(v, fmt=".1f", fallback="k.A."):
    try:
        return f"{float(v):{fmt}}" if pd.notna(v) else fallback
    except Exception:
        return fallback


PRIME_AGE_MIN = 27
PRIME_AGE_MAX = 33
PRIME_FORM_MIN = 0.67
PRIME_STREAK_MIN = 2
PRIME_WINS_MIN = 5
PRIME_AVERAGE_RANGE = "27-33 Jahre"
PRIME_AVERAGE_CENTER = "ca. 30 Jahre"


def get_prime_status(stats):
    age = stats.get('age', np.nan)
    form = stats.get('form', np.nan)
    streak = stats.get('streak', np.nan)
    wins = stats.get('wins', np.nan)

    if any(pd.isna(value) for value in (age, form, streak, wins)):
        return {
            'is_prime': False,
            'badge_html': f'<div class="panel-badge">Prime-Fenster: {PRIME_AVERAGE_RANGE}</div>',
            'reason': (
                f"Prime nicht sicher bestimmbar. Typisch liegt die Prime im Datensatz bei "
                f"{PRIME_AVERAGE_RANGE} mit Zentrum um {PRIME_AVERAGE_CENTER}."
            ),
        }

    age_value = float(age)
    form_value = float(form)
    streak_value = float(streak)
    wins_value = float(wins)

    in_age_window = PRIME_AGE_MIN <= age_value <= PRIME_AGE_MAX
    strong_form = form_value >= PRIME_FORM_MIN
    on_streak = streak_value >= PRIME_STREAK_MIN
    established = wins_value >= PRIME_WINS_MIN
    is_prime = in_age_window and strong_form and on_streak and established

    if is_prime:
        reason = (
            f"Ist in seiner Prime: {safe(age_value, '.1f')} Jahre, Form {form_value * 100:.0f}%, "
            f"Siegesserie von {safe(streak_value, 'g')} und {safe(wins_value, 'g')} UFC-Siegen."
        )
        badge_html = '<div class="panel-badge panel-badge-prime">★ Prime</div>'
    else:
        reason_parts = []
        if not in_age_window:
            reason_parts.append(
                f"Alter {safe(age_value, '.1f')} liegt außerhalb des typischen Prime-Fensters"
            )
        if not strong_form:
            reason_parts.append(f"Form mit {form_value * 100:.0f}% ist aktuell zu schwach")
        if not on_streak:
            reason_parts.append(
                f"Siegesserie ist mit {safe(streak_value, 'g')} noch nicht stark genug"
            )
        if not established:
            reason_parts.append(
                f"mit {safe(wins_value, 'g')} UFC-Siegen ist die Datenbasis noch zu klein"
            )
        reason = "Noch nicht in Prime: " + "; ".join(reason_parts) + "."
        badge_html = f'<div class="panel-badge">Prime-Fenster: {PRIME_AVERAGE_RANGE}</div>'

    reason += (
        f" Im Durchschnitt liegt die Prime hier bei {PRIME_AVERAGE_RANGE}, "
        f"mit Zentrum um {PRIME_AVERAGE_CENTER}."
    )
    return {
        'is_prime': is_prime,
        'badge_html': badge_html,
        'reason': reason,
    }


# ─────────────────────────────────────────────────────────────
# FIGHTER-KARTE HTML
# ─────────────────────────────────────────────────────────────
def fighter_card(name, stats, corner):
    clr = "red" if corner == "red" else "blue"
    photo_path = FIGHTER_PHOTOS.get(name)
    if photo_path and not os.path.isabs(photo_path):
        photo_path = os.path.normpath(os.path.join(os.path.dirname(__file__), photo_path))
    streak     = safe(stats['streak'], "g")
    lng_streak = safe(stats['longest_streak'], "g")
    wins       = safe(stats['wins'], "g")
    losses     = safe(stats['losses'], "g")
    form_pct   = f"{float(stats['form'])*100:.0f}%" if pd.notna(stats['form']) else "k.A."
    height     = f"{float(stats['height']):.0f} cm" if pd.notna(stats['height']) else "k.A."
    reach      = f"{float(stats['reach']):.0f} cm" if pd.notna(stats['reach']) else "k.A."
    strikes    = safe(stats['avg_SIG_STR_landed'], ".1f")
    td         = safe(stats['avg_TD_landed'], ".1f")
    su         = safe(stats['avg_SUB_ATT'], ".2f")
    kd         = safe(stats['avg_KD'], ".2f")
    ctrl       = safe(stats['avg_CTRL_time'], ".0f")

    corner_label = "🔴 ROTE ECKE" if corner == "red" else "🔵 BLAUE ECKE"
    border_color = "#ff4b4b" if corner == "red" else "#4b8bff"

    # Foto anzeigen (lokal via st.image, kein HTML nötig)
    if photo_path and os.path.exists(photo_path):
        st.image(photo_path, width=200)
    else:
        st.markdown(
            f'<div style="height:160px;display:flex;align-items:center;justify-content:center;'
            f'background:#1a1a2e;border-radius:10px;color:#555;font-size:40px;">🥊</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"""
    <div class="fighter-card {clr}-card">
        <div style="font-size:11px;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:4px;">{corner_label}</div>
        <div class="fighter-name">{name}</div>
        <div style="color:#888;font-size:12px;margin-bottom:10px;">UFC Record: {wins}W – {losses}L</div>
        <div class="stat-row"><span class="stat-label">Form (letzte 3)</span><span class="stat-val">{form_pct}</span></div>
        <div class="stat-row"><span class="stat-label">Aktuelle Serie 🔥</span><span class="stat-streak">⚡ {streak} Siege</span></div>
        <div class="stat-row"><span class="stat-label">Längste Serie 🏆</span><span class="stat-streak">🏅 {lng_streak} Siege</span></div>
        <div class="stat-row"><span class="stat-label">Größe</span><span class="stat-val">{height}</span></div>
        <div class="stat-row"><span class="stat-label">Reichweite</span><span class="stat-val">{reach}</span></div>
        <div class="stat-row"><span class="stat-label">Ø Sig. Strikes</span><span class="stat-val">{strikes}</span></div>
        <div class="stat-row"><span class="stat-label">Ø Takedowns</span><span class="stat-val">{td}</span></div>
        <div class="stat-row"><span class="stat-label">Ø Submissions</span><span class="stat-val">{su}</span></div>
        <div class="stat-row"><span class="stat-label">Ø Knockdowns</span><span class="stat-val">{kd}</span></div>
        <div class="stat-row"><span class="stat-label">Ø Bodenkontrolle (s)</span><span class="stat-val">{ctrl}</span></div>
    </div>
    """, unsafe_allow_html=True)


def get_photo_path(name):
    photo_path = FIGHTER_PHOTOS.get(name)
    if photo_path and not os.path.isabs(photo_path):
        photo_path = os.path.normpath(os.path.join(os.path.dirname(__file__), photo_path))
    return photo_path


def fighter_headshot(name):
    photo_path = get_photo_path(name)
    if photo_path and os.path.exists(photo_path):
        st.image(photo_path, width=120)
    else:
        st.markdown(
            '<div style="height:120px;width:120px;display:flex;align-items:center;justify-content:center;'
            'background:#1a1a2e;border-radius:16px;color:#555;font-size:34px;margin:0 auto;">🥊</div>',
            unsafe_allow_html=True
        )


def fighter_panel(name, stats, corner):
    panel_cls = "red-panel" if corner == "red" else "blue-panel"
    corner_label = "Rote Ecke" if corner == "red" else "Blaue Ecke"
    wins = safe(stats['wins'], "g")
    losses = safe(stats['losses'], "g")
    form_pct = f"{float(stats['form'])*100:.0f}%" if pd.notna(stats['form']) else "k.A."
    streak = safe(stats['streak'], "g")
    reach = f"{float(stats['reach']):.0f} cm" if pd.notna(stats['reach']) else "k.A."
    finish_rate = f"{float(stats['finish_rate'])*100:.0f}%" if pd.notna(stats['finish_rate']) else "k.A."
    ko_wins = safe(stats['ko_wins'], "g")
    sub_wins = safe(stats['sub_wins'], "g")
    prime = get_prime_status(stats)

    st.markdown(f"""
    <div class="fighter-panel {panel_cls}">
        <div class="corner-chip">{corner_label}</div>
        <div class="panel-name">{name}</div>
        <div class="panel-record">UFC Record: {wins}W - {losses}L</div>
        <div class="panel-finish">Finish-Rate: {finish_rate} &nbsp;|&nbsp; KO/TKO: {ko_wins} &nbsp;|&nbsp; Subs: {sub_wins}</div>
        <div class="panel-badges">
            <div class="panel-badge">Form: {form_pct}</div>
            <div class="panel-badge">Serie: {streak}</div>
            <div class="panel-badge">Reach: {reach}</div>
            {prime['badge_html']}
        </div>
        <div class="prime-note"><strong>Prime-Status:</strong> {prime['reason']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_tale_of_tape(red_stats, blue_stats):
    def wins_in_last_three(value):
        if pd.notna(value):
            wins = int(round(float(value) * 3))
            return max(0, min(3, wins))
        return None

    def metric_row(label, red_raw, blue_raw, red_display, blue_display):
        left_adv = ""
        right_adv = ""
        label_adv = ""
        if pd.notna(red_raw) and pd.notna(blue_raw):
            red_value = max(0.0, abs(float(red_raw)))
            blue_value = max(0.0, abs(float(blue_raw)))
            total_value = red_value + blue_value
            if total_value > 0:
                red_width = red_value / total_value * 100
                blue_width = blue_value / total_value * 100
            else:
                red_width = blue_width = 50.0
            if float(red_raw) > float(blue_raw):
                left_adv = " tape-adv-left"
                label_adv = " tape-label-adv-left"
            elif float(blue_raw) > float(red_raw):
                right_adv = " tape-adv-right"
                label_adv = " tape-label-adv-right"
        else:
            red_width = blue_width = 12.0

        return (
            f'<div class="tape-row">'
            f'<div class="tape-value-left{left_adv}">{red_display}</div>'
            f'<div class="tape-bar"><div class="tape-fill-red" style="width:{red_width:.1f}%"></div></div>'
            f'<div class="tape-label{label_adv}">{label}</div>'
            f'<div class="tape-bar"><div class="tape-fill-blue" style="width:{blue_width:.1f}%"></div></div>'
            f'<div class="tape-value-right{right_adv}">{blue_display}</div>'
            f'</div>'
        )

    red_recent_wins = wins_in_last_three(red_stats["form"])
    blue_recent_wins = wins_in_last_three(blue_stats["form"])

    rows = [
        metric_row("Siege letzte 3 KÃ¤mpfe", red_stats["form"], blue_stats["form"],
                   f"{red_recent_wins}/3" if red_recent_wins is not None else "k.A.",
                   f"{blue_recent_wins}/3" if blue_recent_wins is not None else "k.A."),
        metric_row("Aktuelle Siegesserie", red_stats["streak"], blue_stats["streak"],
                   safe(red_stats["streak"], "g") + " Siege" if pd.notna(red_stats["streak"]) else "k.A.",
                   safe(blue_stats["streak"], "g") + " Siege" if pd.notna(blue_stats["streak"]) else "k.A."),
        metric_row("Größe", red_stats["height"], blue_stats["height"],
                   f"{float(red_stats['height']):.0f} cm" if pd.notna(red_stats["height"]) else "k.A.",
                   f"{float(blue_stats['height']):.0f} cm" if pd.notna(blue_stats["height"]) else "k.A."),
        metric_row("Reichweite", red_stats["reach"], blue_stats["reach"],
                   f"{float(red_stats['reach']):.0f} cm" if pd.notna(red_stats["reach"]) else "k.A.",
                   f"{float(blue_stats['reach']):.0f} cm" if pd.notna(blue_stats["reach"]) else "k.A."),
        metric_row("Significant Strikes / Kampf", red_stats["avg_SIG_STR_landed"], blue_stats["avg_SIG_STR_landed"],
                   safe(red_stats["avg_SIG_STR_landed"], ".1f") + " / Kampf" if pd.notna(red_stats["avg_SIG_STR_landed"]) else "k.A.",
                   safe(blue_stats["avg_SIG_STR_landed"], ".1f") + " / Kampf" if pd.notna(blue_stats["avg_SIG_STR_landed"]) else "k.A."),
        metric_row("Takedowns / Kampf", red_stats["avg_TD_landed"], blue_stats["avg_TD_landed"],
                   safe(red_stats["avg_TD_landed"], ".1f") + " / Kampf" if pd.notna(red_stats["avg_TD_landed"]) else "k.A.",
                   safe(blue_stats["avg_TD_landed"], ".1f") + " / Kampf" if pd.notna(blue_stats["avg_TD_landed"]) else "k.A."),
        metric_row("Submission-Versuche / Kampf", red_stats["avg_SUB_ATT"], blue_stats["avg_SUB_ATT"],
                   safe(red_stats["avg_SUB_ATT"], ".2f") + " / Kampf" if pd.notna(red_stats["avg_SUB_ATT"]) else "k.A.",
                   safe(blue_stats["avg_SUB_ATT"], ".2f") + " / Kampf" if pd.notna(blue_stats["avg_SUB_ATT"]) else "k.A."),
        metric_row("Knockdowns / Kampf", red_stats["avg_KD"], blue_stats["avg_KD"],
                   safe(red_stats["avg_KD"], ".2f") + " / Kampf" if pd.notna(red_stats["avg_KD"]) else "k.A.",
                   safe(blue_stats["avg_KD"], ".2f") + " / Kampf" if pd.notna(blue_stats["avg_KD"]) else "k.A."),
        metric_row("Kontrollzeit / Kampf", red_stats["avg_CTRL_time"], blue_stats["avg_CTRL_time"],
                   safe(red_stats["avg_CTRL_time"], ".0f") + " s / Kampf" if pd.notna(red_stats["avg_CTRL_time"]) else "k.A.",
                   safe(blue_stats["avg_CTRL_time"], ".0f") + " s / Kampf" if pd.notna(blue_stats["avg_CTRL_time"]) else "k.A."),
    ]

    st.markdown(
        '<div class="tape-card"><div class="tape-title">Tale of the Tape</div>' + "".join(rows) + '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# FEATURE-DIFFERENZEN BERECHNEN
# ─────────────────────────────────────────────────────────────
def build_diffs(rs, bs):
    return {
        'diff_recent_form':        rs['form']              - bs['form'],
        'diff_wins':               rs['wins']              - bs['wins'],
        'diff_current_win_streak': rs['streak']            - bs['streak'],
        'diff_age':                rs['age']               - bs['age'],
        'diff_height':             rs['height']            - bs['height'],
        'diff_reach':              rs['reach']             - bs['reach'],
        'diff_avg_SIG_STR_landed': rs['avg_SIG_STR_landed']- bs['avg_SIG_STR_landed'],
        'diff_avg_TD_landed':      rs['avg_TD_landed']     - bs['avg_TD_landed'],
        'diff_avg_SUB_ATT':        rs['avg_SUB_ATT']       - bs['avg_SUB_ATT'],
        'diff_avg_KD':             rs['avg_KD']            - bs['avg_KD'],
        'diff_avg_CTRL_time':      rs['avg_CTRL_time']     - bs['avg_CTRL_time'],
        'diff_sig_str_defense_proxy': bs['avg_opp_SIG_STR_pct'] - rs['avg_opp_SIG_STR_pct'],
        'diff_td_defense_proxy':   bs['avg_opp_TD_pct']    - rs['avg_opp_TD_pct'],
        'diff_total_rounds_fought': rs['total_rounds_fought'] - bs['total_rounds_fought'],
        'diff_total_time_fought':  rs['total_time_fought'] - bs['total_time_fought'],
        'diff_days_since_last_fight': rs['days_since_last_fight'] - bs['days_since_last_fight'],
    }


# ─────────────────────────────────────────────────────────────
# "WARUM"-ERKLÄRUNG
# ─────────────────────────────────────────────────────────────
def why_explanation(diffs, features, red_name, blue_name, pred, model_name):
    """Zeigt Feature-Beiträge + Textbegründung."""
    red_advantages  = []
    blue_advantages = []

    bars_html = ""
    for feat in features:
        val = diffs.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        label = FEATURE_LABELS.get(feat, feat)
        max_range = max(abs(val), 1)
        pct = min(abs(val) / max_range * 100, 100)
        if val > 0:
            red_advantages.append(label)
            bar = f'<div class="feat-bar-pos" style="width:{pct}%"></div>'
        elif val < 0:
            blue_advantages.append(label)
            bar = f'<div class="feat-bar-neg" style="width:{pct}%"></div>'
        else:
            bar = ""
        display_val = f"{val:+.2f}" if isinstance(val, float) else f"{val:+d}"
        cls = "feat-pos" if val > 0 else "feat-neg"

        bars_html += (
            f'<div class="feat-row">'
            f'<span class="feat-label">{label}</span>'
            f'<div class="feat-bar-wrap">{bar}</div>'
            f'<span class="feat-val {cls}">{display_val}</span>'
            f'</div>'
        )

    winner_name = red_name if pred == 1 else blue_name
    loser_name  = blue_name if pred == 1 else red_name

    adv_text = ""
    if red_advantages:
        adv_text += f"<b>🔴 {red_name}</b> hat Vorteile in: {', '.join(red_advantages[:4])}. "
    if blue_advantages:
        adv_text += f"<b>🔵 {blue_name}</b> hat Vorteile in: {', '.join(blue_advantages[:4])}. "

    adv_feats = red_advantages if pred == 1 else blue_advantages
    feat_list = ', '.join(adv_feats[:3]) if adv_feats else 'k.A.'

    html = (
        '<div class="why-box">'
        f'<div style="font-size:16px;font-weight:700;color:#a78bfa;margin-bottom:8px;">'
        f'🧠 Warum entschied das Modell für {winner_name}?</div>'
        f'<p style="margin-bottom:10px;">'
        f'Das <b>{model_name}</b> bewertet die Differenz aller Features (Rot minus Blau). '
        f'Positive Werte (🔴) begünstigen <b>{red_name}</b>, negative (🔵) <b>{blue_name}</b>.</p>'
        f'{bars_html}'
        f'<p style="margin-top:12px;">{adv_text}</p>'
        f'<p>Das Modell tippt auf <b>{winner_name}</b>, weil dieser Kämpfer in den entscheidenden '
        f'Features (<i>{feat_list}</i>) klare Vorteile zeigt, '
        f'die historisch am stärksten mit einem Sieg korrelieren.</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🥊 UFC Fight Predictor")
    st.caption("KI-gestützte Kampfvorhersage · Modellvergleich · Runden- & Methodenvorhersage · Erklärung")

    with st.spinner("⏳ Lade Daten & trainiere Modelle..."):
        try:
            df, baseline_artifact, advanced_artifact, round_artifact, method_artifact, summary = load_everything()
        except FileNotFoundError as e:
            st.error(f"❌ cleaned_ufc_data.csv nicht gefunden: {e}")
            return

    # ── Modellvergleich-Banner ─────────────────────────────────
    base_metrics = baseline_artifact['metrics']
    adv_metrics = advanced_artifact['metrics']
    acc_base = base_metrics['balanced_accuracy']
    acc_adv = adv_metrics['balanced_accuracy']
    better_model = "Advanced (Random Forest)" if acc_adv >= acc_base else "Baseline (Log. Regression)"
    better_acc = max(acc_adv, acc_base)
    delta = abs(acc_adv - acc_base) * 100

    why_better = (
        f"Der **Random Forest** nutzt {len(ADV_FEATURES)} Features und wird per Zeit-Split bewertet. "
        f"Er liegt in der Balanced Accuracy um {delta:.1f} Prozentpunkte vorn."
        if acc_adv >= acc_base else
        f"Die **Logistische Regression** ist im Zeit-Split robuster und liegt in der Balanced Accuracy "
        f"um {delta:.1f} Prozentpunkte vorn."
    )

    st.markdown(f"""
    <div class="recommend-box">
        <b>🏆 Modell-Empfehlung (automatisch getestet):</b>
        &nbsp;<span style="color:#00e5ff;font-size:16px;">{better_model}</span>
        &nbsp;–&nbsp; Balanced Accuracy: <b>{better_acc*100:.1f}%</b>
        (Baseline: {acc_base*100:.1f}% | Advanced: {acc_adv*100:.1f}%)<br>
        <span style="font-size:13px;opacity:0.85;">{why_better}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Modellauswahl  ─────────────────────────────────────────
    st.markdown("### 🧠 Modellauswahl")
    model_choice = st.radio(
        "Welches Modell soll die Vorhersage treffen?",
        ["Empfehlung: " + better_model, "Baseline (Log. Regression)", "Advanced (Random Forest)"],
        horizontal=True
    )
    selected = "Advanced (Random Forest)" if "Advanced" in model_choice else "Baseline (Log. Regression)"
    winner_artifact = advanced_artifact if selected == "Advanced (Random Forest)" else baseline_artifact
    model_to_use = winner_artifact['model']
    feats_to_use = winner_artifact['features']
    model_label   = "Advanced (Random Forest)"   if selected == "Advanced (Random Forest)" else "Baseline (Logistische Regression)"

    st.markdown("---")

    # ── Kämpfer-Auswahl ───────────────────────────────────────
    fighters = get_fighters(df)
    col_r, col_b = st.columns(2)
    with col_r:
        default_r = fighters.index("Charles Oliveira") if "Charles Oliveira" in fighters else 0
        red_name  = st.selectbox("🔴 Rote Ecke", fighters, index=default_r, key="red")
    with col_b:
        default_b = fighters.index("Ilia Topuria") if "Ilia Topuria" in fighters else 1
        blue_name = st.selectbox("🔵 Blaue Ecke", fighters, index=default_b, key="blue")

    max_data_date = pd.to_datetime(df['date']).max().date()
    default_reference_date = min(date.today(), max_data_date)
    reference_date = st.date_input(
        "Stichtag für den Vergleich",
        value=default_reference_date,
        max_value=max_data_date,
        help="Es werden nur Kämpfe bis zu diesem Datum berücksichtigt. Die Inaktivität wird für beide Kämpfer relativ zu genau diesem Stichtag berechnet."
    )

    if red_name == blue_name:
        st.warning("⚠️ Bitte zwei unterschiedliche Kämpfer wählen!")
        return

    r_stats = get_stats(df, red_name, reference_date=reference_date)
    b_stats = get_stats(df, blue_name, reference_date=reference_date)

    if not r_stats:
        st.error(f"Keine Daten für {red_name} bis zum gewählten Stichtag gefunden."); return
    if not b_stats:
        st.error(f"Keine Daten für {blue_name} bis zum gewählten Stichtag gefunden."); return

    # ── Kämpfer-Karten ─────────────────────────────────────────
    st.markdown("### 🥋 Kämpfer-Profil")
    left_col, center_col, right_col = st.columns([2.2, 1, 2.2])
    with left_col:
        image_col, info_col = st.columns([1, 1.8])
        with image_col:
            fighter_headshot(red_name)
        with info_col:
            fighter_panel(red_name, r_stats, "red")
    with center_col:
        st.markdown("""
        <div class="vs-core">
            <div class="vs-badge">VS</div>
            <div class="vs-caption">Fight Comparison</div>
        </div>
        """, unsafe_allow_html=True)
    with right_col:
        info_col, image_col = st.columns([1.8, 1])
        with info_col:
            fighter_panel(blue_name, b_stats, "blue")
        with image_col:
            fighter_headshot(blue_name)

    render_tale_of_tape(r_stats, b_stats)

    # ── Vorhersage-Button ──────────────────────────────────────
    st.markdown("---")
    if st.button("⚡ Vorhersage berechnen", use_container_width=True):

        diffs = build_diffs(r_stats, b_stats)

        with st.spinner("Modell berechnet…"):
            time.sleep(0.4)
            X_pred = pd.DataFrame([{f: diffs[f] for f in feats_to_use}])
            pred   = model_to_use.predict(X_pred)[0]
            proba  = model_to_use.predict_proba(X_pred)[0]

            prob_red  = proba[1]
            prob_blue = proba[0]

        winner_name = red_name  if pred == 1 else blue_name
        loser_name  = blue_name if pred == 1 else red_name
        corner_cls  = "verdict-red" if pred == 1 else "verdict-blue"
        corner_col  = "#ff4b4b"     if pred == 1 else "#4b8bff"
        bookmaker_context = lookup_bookmaker_context(df, red_name, blue_name, reference_date)
        model_prob = prob_red if pred == 1 else prob_blue

        # ── Ergebnis-Box ──────────────────────────────────────
        st.markdown(f"""
        <div class="verdict-box {corner_cls}">
            <div style="font-size:13px;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:6px;">Vorhersage · {model_label}</div>
            <div class="verdict-name" style="color:{corner_col}">🏆 {winner_name}</div>
            <div style="color:#aaa;font-size:14px;">gewinnt gegen {loser_name}</div>
        </div>
        """, unsafe_allow_html=True)
        render_prediction_comparison(
            winner_name,
            model_prob,
            bookmaker_context,
            bookmaker_context.get("actual_fight"),
            red_name,
            blue_name,
        )

        # ── Wahrscheinlichkeits-Balken ─────────────────────────
        col_pr, col_pb = st.columns(2)
        with col_pr:
            st.metric(f"🔴 {red_name}", f"{prob_red*100:.1f}%")
            st.progress(float(prob_red))
        with col_pb:
            st.metric(f"🔵 {blue_name}", f"{prob_blue*100:.1f}%")
            st.progress(float(prob_blue))

        # ── Runden- & Methoden-Vorhersage ─────────────────────
        st.markdown("#### ⏱️ Kampfverlauf-Vorhersage")
        round_col, method_col = st.columns(2)

        with round_col:
            if round_artifact is not None:
                try:
                    round_model = round_artifact['model'] if round_artifact else None
                    X_round = pd.DataFrame([{f: diffs.get(f, np.nan) for f in round_artifact['features']}])
                    pred_round = round_model.predict(X_round)[0]
                    round_proba = round_model.predict_proba(X_round)[0]
                    rounds_conf = dict(zip(round_model.classes_, round_proba))
                    conf_str = ", ".join([f"R{r}: {p*100:.0f}%" for r,p in sorted(rounds_conf.items())])
                    st.success(f"🔢 **Wahrscheinlichste Runde:** {int(pred_round)}\n\n_{conf_str}_")
                except Exception:
                    st.info("🔢 Runden-Vorhersage nicht verfügbar")
            else:
                st.info("🔢 Runden-Modell nicht trainierbar (fehlende Daten)")

        with method_col:
            if method_artifact is not None:
                try:
                    method_model = method_artifact['model'] if method_artifact else None
                    method_classes = method_model.classes_.tolist() if method_model is not None else []
                    X_method = pd.DataFrame([{f: diffs.get(f, np.nan) for f in method_artifact['features']}])
                    pred_method = method_model.predict(X_method)[0]
                    method_proba = method_model.predict_proba(X_method)[0]
                    method_conf = dict(zip(method_classes, method_proba))
                    conf_str = ", ".join([f"{m}: {p*100:.0f}%" for m,p in sorted(method_conf.items(), key=lambda x:-x[1])])

                    emoji = {"KO / TKO": "💥", "Submission": "🤼", "Decision": "📋"}.get(pred_method, "🏆")
                    st.success(f"{emoji} **Siegmethode:** {pred_method}\n\n_{conf_str}_")
                except Exception:
                    st.info("📋 Methoden-Vorhersage nicht verfügbar")
            else:
                st.info("📋 Methoden-Modell nicht trainierbar (fehlende Daten)")

        # ── Warum-Erklärung ───────────────────────────────────
        st.markdown("---")
        with st.expander("🧠 Warum hat das Modell so entschieden? (Erklärung öffnen)", expanded=True):
            why_explanation(diffs, feats_to_use, red_name, blue_name, pred, model_label)

        # ── Feature Importance (nur Advanced) ─────────────────
        if selected == "Advanced (Random Forest)" and summary.get("advanced_feature_importances"):
            with st.expander("📊 Feature Importance des Advanced Modells"):
                importance_df = pd.DataFrame(
                    list(summary["advanced_feature_importances"].items()),
                    columns=["Feature", "Importance"]
                )
                st.dataframe(importance_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
