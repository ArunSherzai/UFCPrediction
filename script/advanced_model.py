import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def train_and_save_models():
    print("🚀 Starte Training von Baseline und Advanced Modellen...")

    data_path = "../data/cleaned_ufc_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Datei {data_path} nicht gefunden. Bitte zuerst dataCleaning.py ausführen.")
        return

    df = pd.read_csv(data_path)

    # 1. BASELINE MODEL FEATURES (4 Features)
    baseline_features = [
        'diff_recent_form',
        'diff_wins',
        'diff_current_win_streak',
        'diff_age'
    ]

    # 2. ADVANCED MODEL FEATURES (Mehr Features)
    advanced_features = [
        'diff_recent_form',
        'diff_wins',
        'diff_current_win_streak',
        'diff_height',
        'diff_reach',
        'diff_avg_SIG_STR_landed',
        'diff_avg_TD_landed',
        'diff_avg_SUB_ATT',
        'diff_avg_KD',
        'diff_avg_CTRL_time'
    ]

    target = 'target'

    # Filter NaN rows based on all features used so both models are trained on the same data points without NaNs
    all_features_to_check = list(set(advanced_features + baseline_features))
    df_model = df.dropna(subset=all_features_to_check + [target]).copy()

    print(f"📈 Verwendbare Datenpunkte nach NaN-Entfernung: {len(df_model)} (von {len(df)})")

    # Split for Baseline
    X_base = df_model[baseline_features]
    y_base = df_model[target]
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

    # Split for Advanced
    X_adv = df_model[advanced_features]
    y_adv = df_model[target]
    X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # TRAIN BASELINE (Logistic Regression)
    # ---------------------------------------------------------
    print("\n--- Training Baseline Model (Logistic Regression) ---")
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_base, y_train_base)
    base_pred = baseline_model.predict(X_test_base)
    base_acc = accuracy_score(y_test_base, base_pred)
    print(f"🏆 BASELINE ACCURACY: {base_acc:.4f} ({base_acc*100:.2f}%)")

    # ---------------------------------------------------------
    # TRAIN ADVANCED (Random Forest with GridSearchCV)
    # ---------------------------------------------------------
    print("\n--- Training Advanced Model (Random Forest with GridSearchCV) ---")
    rf = RandomForestClassifier(random_state=42)
    
    # GridSearch parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train_adv, y_train_adv)

    best_rf = grid_search.best_estimator_
    print(f"✅ Best Parameters found by GridSearch: {grid_search.best_params_}")

    adv_pred = best_rf.predict(X_test_adv)
    adv_acc = accuracy_score(y_test_adv, adv_pred)
    print(f"🏆 ADVANCED ACCURACY (Random Forest): {adv_acc:.4f} ({adv_acc*100:.2f}%)")

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------
    print("\n--- Feature Importances (Random Forest) ---")
    importances = best_rf.feature_importances_
    feature_names = advanced_features

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    print("Ranking:")
    for f in range(X_adv.shape[1]):
        print(f"{f + 1}. feature {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

    # Optional: Plotting Feature Importances (saves to data folder)
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_adv.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_adv.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, X_adv.shape[1]])
    plt.tight_layout()
    plot_path = "../data/feature_importances.png"
    plt.savefig(plot_path)
    print(f"📊 Feature Importance Plot gespeichert unter {plot_path}")

    # ---------------------------------------------------------
    # SAVE MODELS
    # ---------------------------------------------------------
    os.makedirs("../data", exist_ok=True)
    baseline_path = "../data/baseline_model.pkl"
    advanced_path = "../data/advanced_model.pkl"

    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_model, f)
    with open(advanced_path, 'wb') as f:
        pickle.dump(best_rf, f)

    print(f"\n✅ Beide Modelle ({baseline_path}, {advanced_path}) erfolgreich gespeichert.")

if __name__ == "__main__":
    train_and_save_models()
