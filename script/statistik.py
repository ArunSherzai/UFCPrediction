import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Bereinigte Daten laden
df = pd.read_csv("../data/cleaned_ufc_data.csv")

# 2. Korrelation berechnen
# Wir schauen, wie stark Variablen mit dem Sieg (target) zusammenhängen
# Ein Wert nahe 1 oder -1 bedeutet starken Einfluss, 0 bedeutet egal.
correlations = df.corr(numeric_only=True)['target'].sort_values(ascending=False)

print("--- Top Einflüsse auf den Sieg (Red Corner) ---")
print(correlations.head(10))

# 3. Visualisierung der Sieg-Methoden
plt.figure(figsize=(10, 6))
df['finish_method'].value_counts().plot(kind='bar')
plt.title("Wie werden Kämpfe gewonnen?")
plt.ylabel("Anzahl")
plt.show()