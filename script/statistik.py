import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Bereinigte Daten laden
df = pd.read_csv("../data/cleaned_ufc_data.csv")

# ========================================================
# 2. Korrelation mit dem Sieg (target)
# ========================================================
# Ein Wert nahe 1 oder -1 = starker Einfluss, 0 = kein Einfluss
correlations = df.corr(numeric_only=True)['target'].sort_values(ascending=False)

print("=" * 60)
print("--- Top 15 Einflüsse auf den Sieg (Red Corner) ---")
print("=" * 60)
print(correlations.head(15))
print()
print("--- Bottom 10 (negative Korrelation) ---")
print(correlations.tail(10))

# ========================================================
# 3. Nur die neuen Diff-Features anschauen
# ========================================================
diff_cols = [c for c in df.columns if c.startswith('diff_')]
if diff_cols:
    print()
    print("=" * 60)
    print("--- Korrelation der DIFFERENZ-Features ---")
    print("=" * 60)
    diff_corr = correlations[correlations.index.isin(diff_cols)].sort_values(ascending=False)
    print(diff_corr)

# ========================================================
# 4. Visualisierung: Siegmethoden
# ========================================================
if 'finish_method' in df.columns:
    plt.figure(figsize=(10, 6))
    df['finish_method'].value_counts().plot(kind='bar', color='#e74c3c', edgecolor='black')
    plt.title("Wie werden UFC Kämpfe gewonnen?", fontsize=14, fontweight='bold')
    plt.ylabel("Anzahl")
    plt.xlabel("Siegmethode")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../data/finish_methods.png", dpi=100)
    plt.show()
    print("\n📊 Visualisierung gespeichert: data/finish_methods.png")
else:
    print("\n⚠️ Spalte 'finish_method' nicht vorhanden – Visualisierung übersprungen.")

# ========================================================
# 5. Visualisierung: Top Diff-Features Korrelation
# ========================================================
if diff_cols:
    plt.figure(figsize=(12, 6))
    diff_corr_sorted = correlations[correlations.index.isin(diff_cols)].sort_values(ascending=True)
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in diff_corr_sorted.values]
    diff_corr_sorted.plot(kind='barh', color=colors, edgecolor='black')
    plt.title("Korrelation der Differenz-Features mit dem Sieg", fontsize=14, fontweight='bold')
    plt.xlabel("Korrelation mit Target (1 = Red gewinnt)")
    plt.tight_layout()
    plt.savefig("../data/diff_correlations.png", dpi=100)
    plt.show()
    print("📊 Visualisierung gespeichert: data/diff_correlations.png")