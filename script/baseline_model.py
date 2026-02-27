import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_baseline_model():
    print("🚀 Starte Baseline-Modell Training (Logistische Regression)...")
    
    # Daten laden
    data_path = "../data/cleaned_ufc_data.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Datei {data_path} nicht gefunden. Bitte zuerst dataCleaning.py ausführen.")
        return

    # 3-5 Basis-Features auswählen (da keine Wettquoten im Dataset sind, nutzen wir Form, Siege, Alter)
    features = [
        'diff_recent_form',        # Form der letzten 3 Kämpfe (Siegesrate)
        'diff_wins',               # Differenz der Gesamtsiege
        'diff_current_win_streak', # Aktuelle Sieges-Serie
        'diff_age'                 # Altersunterschied
    ]
    target = 'target' # 1 = Red gewinnt, 0 = Blue gewinnt

    print(f"📊 Verwendete Features: {features}")

    # NaNs entfernen in diesen spezifischen Spalten
    df_model = df.dropna(subset=features + [target])
    
    X = df_model[features]
    y = df_model[target]

    print(f"📈 Verwendbare Datenpunkte nach NaN-Entfernung: {len(df_model)} (von {len(df)})")

    # Train-Test-Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simples Modell ohne tiefes Tuning
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Vorhersagen
    y_pred = model.predict(X_test)

    # Ergebnisse evaluieren
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"🏆 BASELINE ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*40)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # Wichtigkeit der Feature in Logistic Regression (Koeffizienten)
    print("\n--- Feature Weights ---")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    train_baseline_model()
