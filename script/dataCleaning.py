import pandas as pd
import numpy as np
import os


def clean_and_prepare_data():
    """
    Woche 1: Daten laden, bereinigen, Target erstellen.
    Woche 2: Differenzen, Rolling Averages, Label-Encoding.
    """

    # ========================================================
    # 1. PFADE
    # ========================================================
    MAIN_FILE = "../data/data.csv"
    RAW_FIGHT_FILE = "../data/raw_total_fight_data.csv"
    OUTPUT_FILE = "../data/cleaned_ufc_data.csv"

    if not os.path.exists(MAIN_FILE):
        print(f"❌ Fehler: '{MAIN_FILE}' nicht gefunden!")
        return

    # ========================================================
    # 2. LADEN
    # ========================================================
    df = pd.read_csv(MAIN_FILE)
    print(f"✅ Hauptdatei geladen: {len(df)} Kämpfe, {len(df.columns)} Spalten.")

    # win_by aus raw_total_fight_data.csv holen (für Siegmethode pro Kampf)
    if os.path.exists(RAW_FIGHT_FILE):
        raw = pd.read_csv(RAW_FIGHT_FILE, sep=';')
        # Merge: Wir matchen über Kämpfer-Namen und Datum
        # Datumsformate angleichen (raw: "March 20, 2021" → data: "2021-03-20")
        if 'win_by' in raw.columns:
            raw_subset = raw[['R_fighter', 'B_fighter', 'date', 'win_by']].copy()
            raw_subset['date'] = pd.to_datetime(raw_subset['date'], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')
            raw_subset = raw_subset.rename(columns={'win_by': 'finish_method'})
            # Duplikate entfernen (mehrere Runden pro Kampf im raw file)
            raw_subset = raw_subset.drop_duplicates(subset=['R_fighter', 'B_fighter', 'date'])
            df = df.merge(raw_subset, on=['R_fighter', 'B_fighter', 'date'], how='left')
            df = df.copy()  # DataFrame defragmentieren (PerformanceWarning vermeiden)
            print(f"✅ Siegmethode (finish_method) gemergt: {df['finish_method'].notna().sum()} Einträge.")
    else:
        print(f"⚠️ '{RAW_FIGHT_FILE}' nicht gefunden – finish_method wird übersprungen.")

    # ========================================================
    # 3. REINIGUNG (Woche 1)
    # ========================================================
    # Wir brauchen einen Gewinner, sonst kann die KI nichts lernen
    df = df.dropna(subset=['Winner'])

    # Wer hat gewonnen? Red = 1, Blue = 0
    df['target'] = df['Winner'].map({'Red': 1, 'Blue': 0})
    df = df.dropna(subset=['target'])  # Unentschieden/No Contest fliegen raus

    # finish_round als Zahl sicherstellen
    if 'finish_round' in df.columns:
        df['finish_round'] = pd.to_numeric(df['finish_round'], errors='coerce')

    print(f"📊 Nach Reinigung: {len(df)} Kämpfe übrig.")

    # ========================================================
    # 4. WOCHE 2 – DIFFERENZEN ZWISCHEN KÄMPFERN
    # ========================================================
    print("\n--- WOCHE 2: Feature Engineering ---")

    # Alle Differenzen: Red minus Blue
    diff_features = {
        'diff_age':               ('R_age',                    'B_age'),
        'diff_height':            ('R_Height_cms',             'B_Height_cms'),
        'diff_reach':             ('R_Reach_cms',              'B_Reach_cms'),
        'diff_avg_SIG_STR_landed':('R_avg_SIG_STR_landed',     'B_avg_SIG_STR_landed'),
        'diff_avg_SIG_STR_pct':   ('R_avg_SIG_STR_pct',        'B_avg_SIG_STR_pct'),
        'diff_avg_TD_landed':     ('R_avg_TD_landed',           'B_avg_TD_landed'),
        'diff_avg_TD_pct':        ('R_avg_TD_pct',              'B_avg_TD_pct'),
        'diff_avg_SUB_ATT':       ('R_avg_SUB_ATT',             'B_avg_SUB_ATT'),
        'diff_avg_KD':            ('R_avg_KD',                  'B_avg_KD'),
        'diff_avg_CTRL_time':     ('R_avg_CTRL_time(seconds)',   'B_avg_CTRL_time(seconds)'),
        'diff_wins':              ('R_wins',                     'B_wins'),
        'diff_losses':            ('R_losses',                   'B_losses'),
        'diff_current_win_streak':('R_current_win_streak',       'B_current_win_streak'),
        'diff_longest_win_streak':('R_longest_win_streak',       'B_longest_win_streak'),
    }

    created_diffs = []
    for diff_name, (r_col, b_col) in diff_features.items():
        if r_col in df.columns and b_col in df.columns:
            df[diff_name] = df[r_col] - df[b_col]
            created_diffs.append(diff_name)

    print(f"✅ {len(created_diffs)} Differenz-Features erstellt: {', '.join(created_diffs)}")

    # ========================================================
    # 5. WOCHE 2 – ROLLING AVERAGES (Form der letzten 3 Kämpfe)
    # ========================================================
    # Wir sorgen dafür, dass 'date' ein echtes Datum ist
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    def compute_fighter_form(df, fighter_col, is_winner_col='Winner', side='Red', window=3):
        """
        Berechnet für jeden Kämpfer die Form (Rolling Average der letzten `window` Kämpfe).
        Gibt eine Series zurück: Wert zwischen 0.0 (nur Niederlagen) und 1.0 (nur Siege).
        """
        form_values = pd.Series(np.nan, index=df.index, dtype=float)

        # Alle einzigartigen Kämpfer sammeln
        fighters = df[fighter_col].dropna().unique()

        for fighter in fighters:
            # Alle Kämpfe dieses Fighters (in beiden Ecken)
            mask_red = df['R_fighter'] == fighter
            mask_blue = df['B_fighter'] == fighter
            mask_all = mask_red | mask_blue

            fighter_fights = df.loc[mask_all].sort_values('date')

            if len(fighter_fights) < 2:
                continue

            # Hat er gewonnen? (1 = ja, 0 = nein)
            win_series = pd.Series(0, index=fighter_fights.index, dtype=float)
            # Sieg als Red
            win_series[fighter_fights['Winner'] == 'Red'] = (
                fighter_fights.loc[fighter_fights['Winner'] == 'Red', 'R_fighter'] == fighter
            ).astype(float)
            # Sieg als Blue
            win_series[fighter_fights['Winner'] == 'Blue'] = (
                fighter_fights.loc[fighter_fights['Winner'] == 'Blue', 'B_fighter'] == fighter
            ).astype(float)

            # Rolling Average: Nur vergangene Kämpfe (shift, damit der aktuelle nicht mitzählt)
            rolling_form = win_series.shift(1).rolling(window=window, min_periods=1).mean()

            # Nur für die Seite eintragen, auf der der Kämpfer steht
            current_side_mask = df[fighter_col] == fighter
            relevant_idx = fighter_fights.index[fighter_fights.index.isin(df.loc[current_side_mask].index)]
            form_values.loc[relevant_idx] = rolling_form.loc[relevant_idx]

        return form_values

    df['R_recent_form'] = compute_fighter_form(df, 'R_fighter', side='Red', window=3)
    df['B_recent_form'] = compute_fighter_form(df, 'B_fighter', side='Blue', window=3)
    df['diff_recent_form'] = df['R_recent_form'] - df['B_recent_form']

    form_filled = df['R_recent_form'].notna().sum()
    print(f"✅ Rolling Averages berechnet (letzte 3 Kämpfe): {form_filled}/{len(df)} Kämpfer haben Form-Daten.")

    # ========================================================
    # 6. WOCHE 2 – LABEL ENCODING (Kategorien → Zahlen)
    # ========================================================
    # a) finish_method / win_by encodieren
    if 'finish_method' in df.columns:
        finish_mapping = {
            'KO/TKO': 0,
            'Submission': 1,
            'Decision - Unanimous': 2,
            'Decision - Split': 3,
            'Decision - Majority': 4,
            "TKO - Doctor's Stoppage": 5,
            'DQ': 6,
            'Could Not Continue': 7,
            'Overturned': 8,
            'Other': 9
        }
        df['finish_method_encoded'] = df['finish_method'].map(finish_mapping)
        print(f"✅ finish_method encodiert: {df['finish_method_encoded'].notna().sum()} Einträge.")

    # b) Stance encodieren
    stance_mapping = {
        'Orthodox': 0,
        'Southpaw': 1,
        'Switch': 2,
        'Open Stance': 3
    }
    for prefix in ['R_', 'B_']:
        col = f'{prefix}Stance'
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map(stance_mapping)
            print(f"✅ {col} encodiert.")

    # c) weight_class encodieren
    if 'weight_class' in df.columns:
        weight_classes = df['weight_class'].dropna().unique()
        weight_mapping = {wc: i for i, wc in enumerate(sorted(weight_classes))}
        df['weight_class_encoded'] = df['weight_class'].map(weight_mapping)
        print(f"✅ weight_class encodiert: {len(weight_mapping)} Klassen.")

    # d) title_bout zu 0/1
    if 'title_bout' in df.columns:
        df['title_bout'] = df['title_bout'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        print(f"✅ title_bout als 0/1.")

    # ========================================================
    # 7. SPEICHERN
    # ========================================================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*50}")
    print(f"--- ERGEBNIS WOCHE 1 + 2 ---")
    print(f"Datei erstellt: {OUTPUT_FILE}")
    print(f"Kämpfe: {len(df)}")
    print(f"Spalten insgesamt: {len(df.columns)}")

    # Übersicht der neuen Features
    new_cols = [c for c in df.columns if c.startswith('diff_') or 'form' in c.lower() or 'encoded' in c.lower()]
    print(f"\nNeue Woche-2-Features ({len(new_cols)}):")
    for col in sorted(new_cols):
        non_null = df[col].notna().sum()
        print(f"  • {col}: {non_null}/{len(df)} Werte")


if __name__ == "__main__":
    clean_and_prepare_data()