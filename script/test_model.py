import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore') # Um FutureWarnings von sklearn zu unterdrücken

def test_model():
    print("🚀 Lade Modell und Daten...")
    data_path = "../data/cleaned_ufc_data.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Datei {data_path} nicht gefunden. Bitte zuerst dataCleaning.py ausführen.")
        return

    features = ['diff_recent_form', 'diff_wins', 'diff_current_win_streak', 'diff_age']
    target = 'target'
    
    # NaN-Werte für Features, Target und Namens-Spalten entfernen (für konsistente Ausgabe)
    df_model = df.dropna(subset=features + [target, 'R_fighter', 'B_fighter', 'date'])
    
    X = df_model[features]
    y = df_model[target]

    # Modell trainieren (wie im baseline_model.py)
    model = LogisticRegression()
    model.fit(X, y)

    print("✅ Modell erfolgreich trainiert!\n")
    
    while True:
        print("="*50)
        print("🥊 UFC VORHERSAGE - TESTMENÜ")
        print("="*50)
        print("[1] 3 zufällige historische Kämpfe testen")
        print("[2] Eigene Werte eingeben (Experimentieren)")
        print("[3] Beenden")
        print("="*50)
        
        choice = input("\nWähle eine Option (1-3): ")
        
        if choice == '1':
            sample_fights = df_model.sample(3)
            for idx, row in sample_fights.iterrows():
                red = row['R_fighter']
                blue = row['B_fighter']
                date = row['date']
                actual_winner = red if row[target] == 1 else blue
                
                # Input für das Modell extrahieren (als DataFrame um Warnings zu vermeiden)
                X_fight = pd.DataFrame([row[features]], columns=features)
                
                prediction = model.predict(X_fight)[0]
                probabilities = model.predict_proba(X_fight)[0]
                
                predicted_winner = red if prediction == 1 else blue
                prob_red = probabilities[1] * 100
                prob_blue = probabilities[0] * 100
                
                print(f"\n📅 Datum: {date}")
                print(f"🥋 Kampf: {red} (Red) vs. {blue} (Blue)")
                print(f"🤖 Modell-Tipp: {predicted_winner}")
                print(f"📊 Wahrscheinlichkeit: {red} ({prob_red:.1f}%) | {blue} ({prob_blue:.1f}%)")
                print(f"🏆 Echter Sieger: {actual_winner} ", end="")
                
                if predicted_winner == actual_winner:
                    print("- ✅ Richtig!")
                else:
                    print("- ❌ Falsch!")
                    
            print("\n")
            
        elif choice == '2':
            print("\nGib die Differenz-Werte (Team Rot minus Team Blau) ein.")
            print("Beispiel für Alter: Wenn Fighter Red 30 ist und Fighter Blue 28, dann tipp '2' ein.")
            try:
                form_diff = float(input("\n1. Form-Differenz in Siegesrate (-1.0 bis 1.0, z.B. 0.3 für +30% bessere Siegesrate von Red): "))
                win_diff = float(input("2. Differenz der Siege (z.B. 5, wenn Red 5 Siege mehr hat): "))
                streak_diff = float(input("3. Differenz der Siegesserie (z.B. -1, wenn Blue eine bessere Serie hat): "))
                age_diff = float(input("4. Altersunterschied in Jahren (z.B. -3, wenn Red jünger ist): "))
                
                X_custom = pd.DataFrame([[form_diff, win_diff, streak_diff, age_diff]], columns=features)
                pred = model.predict(X_custom)[0]
                probs = model.predict_proba(X_custom)[0]
                
                winner = "Red" if pred == 1 else "Blue"
                print(f"\n🤖 Das Modell tippt auf: => {winner} <=")
                print(f"📊 Wahrscheinlichkeit Red: {probs[1]*100:.1f}% | Blue: {probs[0]*100:.1f}%\n")
                
            except ValueError:
                print("\n❌ Ungültige Eingabe! Bitte nur Zahlen eingeben (Kommastellen mit Punkt eintragen, z.B. '0.5').\n")
                
        elif choice == '3':
            print("👋 Bis zum nächsten Mal!")
            break
        else:
            print("❌ Ungültige Auswahl. Bitte 1, 2 oder 3 eintippen.\n")

if __name__ == "__main__":
    test_model()
