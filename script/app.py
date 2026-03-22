import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import time
import warnings
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
    'diff_days_since_last_fight': 'Inaktivität seit letztem Kampf',
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
    summary_path = os.path.join(base_dir, "..", "data", "model_summary.json")

    if not all(os.path.exists(path) for path in model_paths.values() if "method" not in path and "round" not in path):
        train_all_models(data_path=data_path, output_paths=model_paths)

    baseline_artifact = load_artifact(model_paths["winner_baseline"])
    advanced_artifact = load_artifact(model_paths["winner_advanced"])
    round_artifact = load_artifact(model_paths["fight_round"]) if os.path.exists(model_paths["fight_round"]) else None
    method_artifact = load_artifact(model_paths["fight_method"]) if os.path.exists(model_paths["fight_method"]) else None
    summary = load_model_summary(summary_path)

    return df, baseline_artifact, advanced_artifact, round_artifact, method_artifact, summary


@st.cache_data
def get_fighters(df):
    fighters = set(df['R_fighter'].dropna()).union(set(df['B_fighter'].dropna()))
    return sorted(fighters)


def get_stats(df, name):
    fights = df[(df['R_fighter'] == name) | (df['B_fighter'] == name)].sort_values('date')
    if fights.empty:
        return None
    row = fights.iloc[-1]
    side = 'R_' if row['R_fighter'] == name else 'B_'
    p = side
    return {
        'age':                 row.get(f'{p}age', np.nan),
        'wins':                row.get(f'{p}wins', np.nan),
        'losses':              row.get(f'{p}losses', np.nan),
        'streak':              row.get(f'{p}current_win_streak', np.nan),
        'longest_streak':      row.get(f'{p}longest_win_streak', np.nan),
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
        'days_since_last_fight': row.get(f'{p}days_since_last_fight', np.nan),
    }


def safe(v, fmt=".1f", fallback="k.A."):
    try:
        return f"{float(v):{fmt}}" if pd.notna(v) else fallback
    except Exception:
        return fallback


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
        st.image(photo_path, use_container_width=True)
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

        bars_html += f"""
        <div class="feat-row">
          <span class="feat-label">{label}</span>
          <div class="feat-bar-wrap">{bar}</div>
          <span class="feat-val {'feat-pos' if val>0 else 'feat-neg'}">{display_val}</span>
        </div>"""

    winner_name = red_name if pred == 1 else blue_name
    loser_name  = blue_name if pred == 1 else red_name

    adv_text = ""
    if red_advantages:
        adv_text += f"**🔴 {red_name}** hat Vorteile in: {', '.join(red_advantages[:4])}. "
    if blue_advantages:
        adv_text += f"**🔵 {blue_name}** hat Vorteile in: {', '.join(blue_advantages[:4])}. "

    st.markdown(f"""
    <div class="why-box">
        <div style="font-size:16px;font-weight:700;color:#a78bfa;margin-bottom:8px;">
            🧠 Warum entschied das Modell für {winner_name}?
        </div>
        <p style="margin-bottom:10px;">
            Das <b>{model_name}</b> bewertet die Differenz aller Features (Rot minus Blau).
            Positive Werte (🔴) begünstigen <b>{red_name}</b>, negative (🔵) <b>{blue_name}</b>.
        </p>
        {bars_html}
        <p style="margin-top:12px;">{adv_text}</p>
        <p>Das Modell tippt auf <b>{winner_name}</b>, weil dieser Kämpfer in den entscheidenden
        Features (<i>{', '.join((red_advantages if pred==1 else blue_advantages)[:3])}</i>) klare Vorteile zeigt,
        die historisch am stärksten mit einem Sieg korrelieren.</p>
    </div>
    """, unsafe_allow_html=True)


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

    if red_name == blue_name:
        st.warning("⚠️ Bitte zwei unterschiedliche Kämpfer wählen!")
        return

    r_stats = get_stats(df, red_name)
    b_stats = get_stats(df, blue_name)

    if not r_stats:
        st.error(f"Keine Daten für {red_name} gefunden."); return
    if not b_stats:
        st.error(f"Keine Daten für {blue_name} gefunden."); return

    # ── Kämpfer-Karten ─────────────────────────────────────────
    st.markdown("### 🥋 Kämpfer-Profil")
    col1, col2 = st.columns(2)
    with col1: fighter_card(red_name,  r_stats, "red")
    with col2: fighter_card(blue_name, b_stats, "blue")

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

        # ── Ergebnis-Box ──────────────────────────────────────
        st.markdown(f"""
        <div class="verdict-box {corner_cls}">
            <div style="font-size:13px;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:6px;">Vorhersage · {model_label}</div>
            <div class="verdict-name" style="color:{corner_col}">🏆 {winner_name}</div>
            <div style="color:#aaa;font-size:14px;">gewinnt gegen {loser_name}</div>
        </div>
        """, unsafe_allow_html=True)

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
