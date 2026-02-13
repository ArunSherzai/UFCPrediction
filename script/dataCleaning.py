import pandas as pd
import os


def clean_and_prepare_data():
    # 1. PFADE (Exakt so, wie sie in deinem PyCharm-Ordner heißen)
    MAIN_FILE = "../data/data.csv"
    DETAILS_FILE = "../data/row_fighter_details.csv"  # Dein Dateiname aus dem Screenshot
    OUTPUT_FILE = "../data/cleaned_ufc_data.csv"

    # Prüfen, ob die Dateien da sind
    if not os.path.exists(MAIN_FILE):
        print(f"❌ Fehler: '{MAIN_FILE}' nicht gefunden. Prüfe den Namen im Ordner!")
        return

    # 2. LADEN
    df = pd.read_csv(MAIN_FILE)
    print(f"✅ Hauptdatei geladen: {len(df)} Kämpfe.")

    # 3. REINIGUNG (Target Definition)
    # Wir brauchen einen Gewinner, sonst kann die KI nichts lernen
    df = df.dropna(subset=['Winner'])

    # Wer hat gewonnen? Red = 1, Blue = 0
    df['target'] = df['Winner'].map({'Red': 1, 'Blue': 0})
    df = df.dropna(subset=['target'])  # Unentschieden/No Contest fliegen raus

    # 4. ZIELVARIABLEN VORBEREITEN (Für deine Predictions)
    # Wir stellen sicher, dass die Runde eine Zahl ist
    if 'finish_round' in df.columns:
        df['finish_round'] = pd.to_numeric(df['finish_round'], errors='coerce')

    # 5. ERSTES FEATURE ENGINEERING (Differenzen)
    # Hier nutzen wir die Spalten, die in der 'data.csv' meist schon drin sind
    # Wir berechnen den Altersunterschied (Red - Blue)
    if 'R_age' in df.columns and 'B_age' in df.columns:
        df['diff_age'] = df['R_age'] - df['B_age']
        print("📊 Feature erstellt: Altersdifferenz")

    # 6. SPEICHERN
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- ERGEBNIS WOCHE 1 ---")
    print(f"Datei erfolgreich erstellt: {OUTPUT_FILE}")
    print(f"Spalten insgesamt: {len(df.columns)}")


if __name__ == "__main__":
    clean_and_prepare_data()