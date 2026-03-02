import streamlit as st
import pandas as pd
import pickle
import os
import time
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="centered")

# Custom CSS für eine schönere UI (die Boxen für Fighter)
st.markdown("""
    <style>
    .fighter-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #2b2b2b;
        color: white;
        margin-bottom: 20px;
    }
    .red-corner {
        border-left: 8px solid #ff4b4b;
    }
    .blue-corner {
        border-left: 8px solid #4b8bff;
    }
    .stat-label {
        font-size: 14px;
        color: #aaaaaa;
    }
    .stat-value {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "cleaned_ufc_data.csv")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Feature Engineering fürs Training
    features = ['diff_recent_form', 'diff_wins', 'diff_current_win_streak', 'diff_age']
    target = 'target'
    df_model = df.dropna(subset=features + [target])
    
    baseline_model = None
    advanced_model = None
    
    baseline_model_path = os.path.join(base_dir, "..", "data", "baseline_model.pkl")
    if os.path.exists(baseline_model_path):
        with open(baseline_model_path, "rb") as f:
            baseline_model = pickle.load(f)
            
    advanced_model_path = os.path.join(base_dir, "..", "data", "advanced_model.pkl")
    if os.path.exists(advanced_model_path):
        with open(advanced_model_path, "rb") as f:
            advanced_model = pickle.load(f)
            
    # Model Setup Fallback
    if not baseline_model:
        X = df_model[features]
        y = df_model[target]
        baseline_model = LogisticRegression(max_iter=1000)
        baseline_model.fit(X, y)
    
    return df, baseline_model, advanced_model, df_model

@st.cache_data
def get_fighters_list(df):
    fighters = set(df['R_fighter'].dropna()).union(set(df['B_fighter'].dropna()))
    return sorted(list(fighters))

def get_latest_stats(df, fighter_name):
    # Alle Kämpfe dieses Kämpfers
    fights = df[(df['R_fighter'] == fighter_name) | (df['B_fighter'] == fighter_name)].copy()
    if fights.empty:
        return None
    
    # Sicherstellen, dass sie nach Datum sortiert sind (ältester -> neuester)
    fights = fights.sort_values(by='date')
    
    # Der aktuellste Kampf
    last_fight = fights.iloc[-1]
    
    stats = {}
    if last_fight['R_fighter'] == fighter_name:
        stats['age'] = last_fight['R_age']
        stats['wins'] = last_fight['R_wins']
        stats['streak'] = last_fight['R_current_win_streak']
        stats['form'] = last_fight['R_recent_form']
        stats['height'] = last_fight['R_Height_cms']
        stats['reach'] = last_fight['R_Reach_cms']
        stats['avg_SIG_STR_landed'] = last_fight['R_avg_SIG_STR_landed']
        stats['avg_TD_landed'] = last_fight['R_avg_TD_landed']
        stats['avg_SUB_ATT'] = last_fight['R_avg_SUB_ATT']
        stats['avg_KD'] = last_fight['R_avg_KD']
        stats['avg_CTRL_time'] = last_fight['R_avg_CTRL_time(seconds)']
    else:
        stats['age'] = last_fight['B_age']
        stats['wins'] = last_fight['B_wins']
        stats['streak'] = last_fight['B_current_win_streak']
        stats['form'] = last_fight['B_recent_form']
        stats['height'] = last_fight['B_Height_cms']
        stats['reach'] = last_fight['B_Reach_cms']
        stats['avg_SIG_STR_landed'] = last_fight['B_avg_SIG_STR_landed']
        stats['avg_TD_landed'] = last_fight['B_avg_TD_landed']
        stats['avg_SUB_ATT'] = last_fight['B_avg_SUB_ATT']
        stats['avg_KD'] = last_fight['B_avg_KD']
        stats['avg_CTRL_time'] = last_fight['B_avg_CTRL_time(seconds)']
        
    return stats

def main():
    st.title("🥊 UFC Fight Predictor")
    st.markdown("Wähle zwei Kämpfer aus und lass das Modell den wahrscheinlichen Gewinner berechnen!")
    
    with st.spinner("Lade Daten und Modelle..."):
        try:
            df, baseline_model, advanced_model, df_model = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ Datei 'cleaned_ufc_data.csv' nicht gefunden: {e}")
            return

    # Model Selection UI
    st.markdown("### 🧠 Modellauswahl")
    model_choice = st.radio(
        "Wähle das Modell für die Vorhersage:",
        ["Baseline (Logistische Regression)", "Advanced (Random Forest)"],
        horizontal=True
    )
    
    if model_choice == "Advanced (Random Forest)" and not advanced_model:
        st.warning("⚠️ Advanced Modell nicht gefunden. Bitte trainiere es zuerst mit `advanced_model.py`.")
        model_choice = "Baseline (Logistische Regression)"

    st.markdown("---")

    fighters = get_fighters_list(df)
    
    # Layout für die Auswahl
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Rote Ecke (Favorit)")
        red_fighter = st.selectbox("Kämpfer wählen", fighters, index=fighters.index('Conor McGregor') if 'Conor McGregor' in fighters else 0, key='red')
        
    with col2:
        st.subheader("🔵 Blaue Ecke (Herausforderer)")
        blue_fighter = st.selectbox("Kämpfer wählen", fighters, index=fighters.index('Khabib Nurmagomedov') if 'Khabib Nurmagomedov' in fighters else 1, key='blue')
        
    if red_fighter == blue_fighter:
        st.warning("⚠️ Bitte wähle zwei unterschiedliche Kämpfer!")
        return
        
    st.markdown("---")
    
    red_stats = get_latest_stats(df, red_fighter)
    blue_stats = get_latest_stats(df, blue_fighter)
    
    if not red_stats or pd.isna(red_stats['age']) or pd.isna(red_stats['form']):
        st.error(f"Nicht genug Datenpunkte für {red_fighter}, um alle Features zu extrahieren.")
        return
    if not blue_stats or pd.isna(blue_stats['age']) or pd.isna(blue_stats['form']):
        st.error(f"Nicht genug Datenpunkte für {blue_fighter}, um alle Features zu extrahieren.")
        return

    # Darstellung der Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="fighter-box red-corner">
            <h3>{red_fighter}</h3>
            <div class="stat-label">Alter</div>
            <div class="stat-value">{int(red_stats['age'])}</div>
            <div class="stat-label">Gesamtsiege (vor letztem Kampf)</div>
            <div class="stat-value">{int(red_stats['wins'])}</div>
            <div class="stat-label">Aktuelle Siegesserie</div>
            <div class="stat-value">{int(red_stats['streak'])}</div>
            <div class="stat-label">Form (Letzte 3 Kämpfe)</div>
            <div class="stat-value">{red_stats['form']*100:.0f}% Winrate</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="fighter-box blue-corner">
            <h3>{blue_fighter}</h3>
            <div class="stat-label">Alter</div>
            <div class="stat-value">{int(blue_stats['age'])}</div>
            <div class="stat-label">Gesamtsiege (vor letztem Kampf)</div>
            <div class="stat-value">{int(blue_stats['wins'])}</div>
            <div class="stat-label">Aktuelle Siegesserie</div>
            <div class="stat-value">{int(blue_stats['streak'])}</div>
            <div class="stat-label">Form (Letzte 3 Kämpfe)</div>
            <div class="stat-value">{blue_stats['form']*100:.0f}% Winrate</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Feature Differenzen berechnen
    diff_form = red_stats['form'] - blue_stats['form']
    diff_wins = red_stats['wins'] - blue_stats['wins']
    diff_streak = red_stats['streak'] - blue_stats['streak']
    diff_age = red_stats['age'] - blue_stats['age']
    diff_height = red_stats['height'] - blue_stats['height']
    diff_reach = red_stats['reach'] - blue_stats['reach']
    diff_avg_SIG_STR_landed = red_stats['avg_SIG_STR_landed'] - blue_stats['avg_SIG_STR_landed']
    diff_avg_TD_landed = red_stats['avg_TD_landed'] - blue_stats['avg_TD_landed']
    diff_avg_SUB_ATT = red_stats['avg_SUB_ATT'] - blue_stats['avg_SUB_ATT']
    diff_avg_KD = red_stats['avg_KD'] - blue_stats['avg_KD']
    diff_avg_CTRL_time = red_stats['avg_CTRL_time'] - blue_stats['avg_CTRL_time']

    # NA Handling für Advanced Features
    X_pred_adv_dict = {
        'diff_recent_form': [diff_form],
        'diff_wins': [diff_wins],
        'diff_current_win_streak': [diff_streak],
        'diff_height': [diff_height],
        'diff_reach': [diff_reach],
        'diff_avg_SIG_STR_landed': [diff_avg_SIG_STR_landed],
        'diff_avg_TD_landed': [diff_avg_TD_landed],
        'diff_avg_SUB_ATT': [diff_avg_SUB_ATT],
        'diff_avg_KD': [diff_avg_KD],
        'diff_avg_CTRL_time': [diff_avg_CTRL_time]
    }
    
    # Button für Prediction
    if st.button("Vorhersage berechnen 🎲", use_container_width=True):
        
        # Check for NaNs before prediction
        if model_choice == "Advanced (Random Forest)":
            # Advanced needs all advanced features
            has_nan = any(pd.isna(val[0]) for val in X_pred_adv_dict.values())
            if has_nan:
                st.error("Nicht ausreichend Datenpunkte (Größe, Reichweite oder spezifische Kampf-Metriken) für die gewählten Kämpfer vorhanden, um das Advanced Modell anzuwenden. Bitte wähle andere Kämpfer oder wechsle zum Baseline Modell.")
                return
        
        with st.spinner("Modell berechnet..."):
            time.sleep(0.5) # Kleiner Delay für dramatischeren Effekt
            
            if model_choice == "Baseline (Logistische Regression)":
                X_pred = pd.DataFrame({
                    'diff_recent_form': [diff_form],
                    'diff_wins': [diff_wins],
                    'diff_current_win_streak': [diff_streak],
                    'diff_age': [diff_age]
                })
                model_to_use = baseline_model
                acc_info = "Baseline Modell mit 4 Features (Accuracy auf Testdaten: ~64%)."
            else:
                X_pred = pd.DataFrame(X_pred_adv_dict)
                model_to_use = advanced_model
                acc_info = "Advanced Modell (Random Forest) mit 10 Features (ohne Alter, Accuracy auf Testdaten: ~64%)."
            
            prediction = model_to_use.predict(X_pred)[0]
            probabilities = model_to_use.predict_proba(X_pred)[0]
            
            prob_blue = probabilities[0]
            prob_red = probabilities[1]
            
            st.markdown(f"### 📊 Vorhersage-Ergebnis ({model_choice})")
            
            if prediction == 1:
                st.success(f"🏆 Das Modell sagt voraus: **{red_fighter} (Red Corner) gewinnt!**")
            else:
                st.info(f"🏆 Das Modell sagt voraus: **{blue_fighter} (Blue Corner) gewinnt!**")
                
            st.markdown(f"**Gewinnwahrscheinlichkeit Rote Ecke ({red_fighter}):** {prob_red*100:.1f}%")
            st.progress(float(prob_red))
            
            st.markdown(f"**Gewinnwahrscheinlichkeit Blaue Ecke ({blue_fighter}):** {prob_blue*100:.1f}%")
            st.progress(float(prob_blue))
            
            st.caption(f"{acc_info} Trainiert auf {len(df_model)} Matches.")
            
            # Feature Importance Show (only for Advanced)
            base_dir = os.path.dirname(__file__)
            importance_path = os.path.join(base_dir, "..", "data", "feature_importances.png")
            if model_choice == "Advanced (Random Forest)" and os.path.exists(importance_path):
                with st.expander("Warum hat das Modell so entschieden? (Feature Importance ansehen)"):
                    st.image(importance_path, caption="Wichtigkeit der verschiedenen Statistiken für das Advanced Modell", use_container_width=True)

if __name__ == "__main__":
    main()
