import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')  # Um FutureWarnings von sklearn zu unterdrücken


# ──────────────────────────────────────────────
# HILFSFUNKTIONEN
# ──────────────────────────────────────────────

def load_data(data_path="../data/cleaned_ufc_data.csv"):
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print(f"❌ Datei {data_path} nicht gefunden. Bitte zuerst dataCleaning.py ausführen.")
        return None


def format_finish(row):
    """Gibt einen lesbaren String für Siegmethode + Runde zurück."""
    method = row.get('finish_method', None)
    runde = row.get('finish_round', None)

    parts = []
    if pd.notna(method) and str(method).strip() not in ('', 'nan'):
        parts.append(str(method).strip())
    if pd.notna(runde):
        parts.append(f"Runde {int(runde)}")

    return " | ".join(parts) if parts else "k.A."


# ──────────────────────────────────────────────
# MODELL: SIEGER-VORHERSAGE (Baseline)
# ──────────────────────────────────────────────

WIN_FEATURES = ['diff_recent_form', 'diff_wins', 'diff_current_win_streak', 'diff_age']


def build_winner_model(df):
    target = 'target'
    df_model = df.dropna(subset=WIN_FEATURES + [target])
    X = df_model[WIN_FEATURES]
    y = df_model[target]
    model = LogisticRegression()
    model.fit(X, y)
    return model, df_model


# ──────────────────────────────────────────────
# MODELL: RUNDENVORHERSAGE
# ──────────────────────────────────────────────

ROUND_FEATURES = [
    'diff_recent_form',
    'diff_wins',
    'diff_current_win_streak',
    'diff_avg_SIG_STR_landed',
    'diff_avg_TD_landed',
    'diff_avg_SUB_ATT',
    'diff_avg_KD',
    'diff_avg_CTRL_time',
]


def build_round_model(df, model_type='separate'):
    """
    Baut ein Modell zur Vorhersage der Kampfrunde (finish_round).

    model_type:
        'separate'  – Eigenes RandomForest-Modell nur für die Runde
        'combined'  – Nutzt Sieger-Features (4 Features) für die Runde
    """
    if 'finish_round' not in df.columns:
        print("⚠️ Spalte 'finish_round' nicht im Datensatz vorhanden.")
        return None, None, None

    target_round = 'finish_round'

    if model_type == 'separate':
        features = ROUND_FEATURES
    else:  # combined
        features = WIN_FEATURES

    df_round = df.dropna(subset=features + [target_round]).copy()
    df_round[target_round] = df_round[target_round].astype(int)

    if len(df_round) < 50:
        print("⚠️ Zu wenige Datenpunkte für Rundenvorhersage.")
        return None, None, None

    X = df_round[features]
    y = df_round[target_round]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return rf, features, acc


def compare_round_models(df):
    """Vergleicht separates vs. kombiniertes Rundenvorhersagemodell."""
    print("\n" + "═"*55)
    print("  🔬 MODELL-VERGLEICH: Rundenvorhersage")
    print("═"*55)

    rf_sep, feats_sep, acc_sep = build_round_model(df, model_type='separate')
    rf_com, feats_com, acc_com = build_round_model(df, model_type='combined')

    if rf_sep is None or rf_com is None:
        print("❌ Vergleich nicht möglich (zu wenige Daten).")
        return None, None

    print(f"\n📊 Separates Modell  (erweiterte Features):  Accuracy = {acc_sep*100:.2f}%")
    print(f"📊 Kombiniertes Modell (Sieger-Features):     Accuracy = {acc_com*100:.2f}%")

    if acc_sep >= acc_com:
        print(f"\n✅ Empfehlung: Das SEPARATE Modell performt besser (+{(acc_sep-acc_com)*100:.2f}%).")
        best_rf, best_feats = rf_sep, feats_sep
    else:
        print(f"\n✅ Empfehlung: Das KOMBINIERTE Modell performt besser (+{(acc_com-acc_sep)*100:.2f}%).")
        best_rf, best_feats = rf_com, feats_com

    print("═"*55)
    return best_rf, best_feats


# ──────────────────────────────────────────────
# HAUPTMENÜ
# ──────────────────────────────────────────────

def test_model():
    print("🚀 Lade Modell und Daten...")
    df = load_data()
    if df is None:
        return

    winner_model, df_model = build_winner_model(df)
    print("✅ Sieger-Modell erfolgreich trainiert!\n")

    # Bestes Rundenmodell (Vergleich beim Start)
    round_model, round_features = compare_round_models(df)

    while True:
        print("\n" + "="*50)
        print("🥊 UFC VORHERSAGE - TESTMENÜ")
        print("="*50)
        print("[1] 3 zufällige historische Kämpfe testen")
        print("[2] Eigene Werte eingeben (Experimentieren)")
        print("[3] Modell-Vergleich Rundenvorhersage (erneut)")
        print("[4] Beenden")
        print("="*50)

        choice = input("\nWähle eine Option (1-4): ").strip()

        # ── Option 1: Historische Kämpfe ──────────────────────
        if choice == '1':
            required_cols = WIN_FEATURES + ['target', 'R_fighter', 'B_fighter', 'date']
            sample_df = df_model.dropna(subset=required_cols)
            sample_fights = sample_df.sample(3)

            for idx, row in sample_fights.iterrows():
                red   = row['R_fighter']
                blue  = row['B_fighter']
                date  = row['date']
                actual_winner = red if row['target'] == 1 else blue

                X_fight = pd.DataFrame([row[WIN_FEATURES]], columns=WIN_FEATURES)

                prediction    = winner_model.predict(X_fight)[0]
                probabilities = winner_model.predict_proba(X_fight)[0]

                predicted_winner = red if prediction == 1 else blue
                prob_red  = probabilities[1] * 100
                prob_blue = probabilities[0] * 100

                # Runden-Vorhersage (wenn Modell verfügbar)
                round_pred_str = "k.A."
                if round_model is not None:
                    try:
                        X_round = pd.DataFrame([row[round_features]], columns=round_features)
                        predicted_round = round_model.predict(X_round)[0]
                        round_pred_str = f"Runde {int(predicted_round)}"
                    except Exception:
                        pass

                # Echter Ausgang
                finish_str = format_finish(row)

                print(f"\n📅 Datum:            {date}")
                print(f"🥋 Kampf:            {red} (Red) vs. {blue} (Blue)")
                print(f"🤖 Modell-Tipp:      {predicted_winner}")
                print(f"📊 Wahrscheinlichkeit: {red} ({prob_red:.1f}%) | {blue} ({prob_blue:.1f}%)")
                print(f"🔢 Runden-Vorhersage: {round_pred_str}")
                print(f"📋 Echter Ausgang:   {actual_winner}  [{finish_str}]", end="  ")

                if predicted_winner == actual_winner:
                    print("✅ Richtig!")
                else:
                    print("❌ Falsch!")

            print()

        # ── Option 2: Eigene Werte ────────────────────────────
        elif choice == '2':
            print("\nGib die Differenz-Werte (Team Rot minus Team Blau) ein.")
            print("Beispiel: Wenn Red 30 Jahre alt und Blue 28 → tipp '2' ein.")
            try:
                form_diff   = float(input("\n1. Form-Differenz (-1.0 bis 1.0): "))
                win_diff    = float(input("2. Differenz der Siege: "))
                streak_diff = float(input("3. Differenz der Siegesserie: "))
                age_diff    = float(input("4. Altersunterschied in Jahren: "))

                X_custom = pd.DataFrame(
                    [[form_diff, win_diff, streak_diff, age_diff]],
                    columns=WIN_FEATURES
                )
                pred  = winner_model.predict(X_custom)[0]
                probs = winner_model.predict_proba(X_custom)[0]

                winner = "Red" if pred == 1 else "Blue"
                print(f"\n🤖 Das Modell tippt auf: => {winner} <=")
                print(f"📊 Wahrscheinlichkeit Red: {probs[1]*100:.1f}% | Blue: {probs[0]*100:.1f}%\n")

            except ValueError:
                print("\n❌ Ungültige Eingabe! Bitte nur Zahlen eingeben (Punkt statt Komma, z.B. '0.5').\n")

        # ── Option 3: Modell-Vergleich ────────────────────────
        elif choice == '3':
            round_model, round_features = compare_round_models(df)

        # ── Option 4: Beenden ─────────────────────────────────
        elif choice == '4':
            print("👋 Bis zum nächsten Mal!")
            break

        else:
            print("❌ Ungültige Auswahl. Bitte 1, 2, 3 oder 4 eintippen.\n")


if __name__ == "__main__":
    test_model()
