import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

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
    df = pd.read_csv("../data/cleaned_ufc_data.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Feature Engineering fürs Training
    features = ['diff_recent_form', 'diff_wins', 'diff_current_win_streak', 'diff_age']
    target = 'target'
    df_model = df.dropna(subset=features + [target])
    
    X = df_model[features]
    y = df_model[target]
    
    # Model Setup
    model = LogisticRegression()
    model.fit(X, y)
    
    return df, model, df_model

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
    else:
        stats['age'] = last_fight['B_age']
        stats['wins'] = last_fight['B_wins']
        stats['streak'] = last_fight['B_current_win_streak']
        stats['form'] = last_fight['B_recent_form']
        
    return stats

def main():
    st.title("🥊 UFC Fight Predictor (Baseline)")
    st.markdown("Wähle zwei Kämpfer aus und lass das Logistische-Regression-Modell den wahrscheinlichen Gewinner berechnen!")
    
    with st.spinner("Lade Daten und trainiere Baseline-Modell..."):
        try:
            df, model, df_model = load_data()
        except FileNotFoundError:
            st.error("❌ Datei ../data/cleaned_ufc_data.csv nicht gefunden. Bitte zuerst dataCleaning.py ausführen.")
            return

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
    
    X_pred = pd.DataFrame([[diff_form, diff_wins, diff_streak, diff_age]], columns=['diff_recent_form', 'diff_wins', 'diff_current_win_streak', 'diff_age'])
    
    # Button für Prediction
    if st.button("Vorhersage berechnen 🎲", use_container_width=True):
        with st.spinner("Modell berechnet..."):
            time.sleep(0.5) # Kleiner Delay für dramatischeren Effekt
            
            prediction = model.predict(X_pred)[0]
            probabilities = model.predict_proba(X_pred)[0]
            
            prob_red = probabilities[1]
            prob_blue = probabilities[0]
            
            st.markdown("### 📊 Vorhersage-Ergebnis")
            
            if prediction == 1:
                st.success(f"🏆 Das Modell sagt voraus: **{red_fighter} (Red Corner) gewinnt!**")
            else:
                st.info(f"🏆 Das Modell sagt voraus: **{blue_fighter} (Blue Corner) gewinnt!**")
                
            st.markdown(f"**Gewinnwahrscheinlichkeit Rote Ecke ({red_fighter}):** {prob_red*100:.1f}%")
            st.progress(float(prob_red))
            
            st.markdown(f"**Gewinnwahrscheinlichkeit Blaue Ecke ({blue_fighter}):** {prob_blue*100:.1f}%")
            st.progress(float(prob_blue))
            
            st.caption(f"Baseline Modell mit 4 Features (Accuracy auf Testdaten: ~66%). Trainiert auf {len(df_model)} Matches.")

if __name__ == "__main__":
    main()
