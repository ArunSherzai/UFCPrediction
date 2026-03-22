import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Kein GUI-Fenster nötig

# 1. Daten laden
df = pd.read_csv("../data/cleaned_ufc_data.csv")

# 2. Alle numerischen Korrelationen mit dem Target
correlations = df.corr(numeric_only=True)['target'].drop('target').sort_values(ascending=False)

# ========================================================
# CHART 1: Top 20 wichtigste Features (positiv + negativ)
# ========================================================
top_pos = correlations.head(20)
top_neg = correlations.tail(10)
top_features = pd.concat([top_pos, top_neg]).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors = []
for v in top_features.values:
    if abs(v) >= 0.08:
        colors.append('#2ecc71' if v > 0 else '#e74c3c')  # Grün/Rot = wichtig
    elif abs(v) >= 0.04:
        colors.append('#f39c12')  # Orange = mittel
    else:
        colors.append('#95a5a6')  # Grau = unwichtig

ax.barh(range(len(top_features)), top_features.values, color=colors, edgecolor='#2c3e50', linewidth=0.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features.index, fontsize=10)
ax.set_xlabel('Korrelation mit Sieg (Red Corner)', fontsize=12)
ax.set_title('Top 30 Features: Welche beeinflussen den Sieg?', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.axvline(x=0.08, color='#2ecc71', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=-0.08, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5)

# Legende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='#2c3e50', label='Stark positiv (≥ 0.08)'),
    Patch(facecolor='#e74c3c', edgecolor='#2c3e50', label='Stark negativ (≤ -0.08)'),
    Patch(facecolor='#f39c12', edgecolor='#2c3e50', label='Mittelmäßig (0.04–0.08)'),
    Patch(facecolor='#95a5a6', edgecolor='#2c3e50', label='Kaum Einfluss (< 0.04)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig("../data/feature_importance_top30.png", dpi=150, bbox_inches='tight')
print("✅ Chart 1 gespeichert: data/feature_importance_top30.png")

# ========================================================
# CHART 2: Nur die DIFF-Features (Woche 2)
# ========================================================
diff_cols = [c for c in correlations.index if c.startswith('diff_')]
diff_corr = correlations[diff_cols].sort_values(ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 7))
colors2 = []
for v in diff_corr.values:
    if abs(v) >= 0.08:
        colors2.append('#2ecc71' if v > 0 else '#e74c3c')
    elif abs(v) >= 0.04:
        colors2.append('#f39c12')
    else:
        colors2.append('#95a5a6')

bars = ax2.barh(range(len(diff_corr)), diff_corr.values, color=colors2, edgecolor='#2c3e50', linewidth=0.5)
ax2.set_yticks(range(len(diff_corr)))
ax2.set_yticklabels(diff_corr.index, fontsize=11)
ax2.set_xlabel('Korrelation mit Sieg (Red Corner)', fontsize=12)
ax2.set_title('Woche 2: Kämpfer-Differenzen – Was zählt wirklich?', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.8)

# Wert neben jeden Balken schreiben
for i, (val, name) in enumerate(zip(diff_corr.values, diff_corr.index)):
    ax2.text(val + 0.003 if val >= 0 else val - 0.003,
             i, f'{val:.3f}',
             va='center', ha='left' if val >= 0 else 'right',
             fontsize=9, fontweight='bold')

ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("../data/diff_feature_importance.png", dpi=150, bbox_inches='tight')
print("✅ Chart 2 gespeichert: data/diff_feature_importance.png")

# ========================================================
# CHART 3: Heatmap – Form & Streaks Zusammenhang
# ========================================================
form_cols = ['R_recent_form', 'B_recent_form', 'diff_recent_form',
             'R_current_win_streak', 'B_current_win_streak', 'diff_current_win_streak',
             'R_longest_win_streak', 'B_longest_win_streak', 'diff_longest_win_streak',
             'target']
form_cols = [c for c in form_cols if c in df.columns]

fig3, ax3 = plt.subplots(figsize=(10, 8))
corr_matrix = df[form_cols].corr()
im = ax3.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-0.3, vmax=0.3, aspect='auto')

ax3.set_xticks(range(len(form_cols)))
ax3.set_yticks(range(len(form_cols)))
ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
ax3.set_yticklabels(corr_matrix.index, fontsize=9)
ax3.set_title('Form & Siegesserien – Korrelationsmatrix', fontsize=14, fontweight='bold')

# Werte in die Zellen schreiben
for i in range(len(form_cols)):
    for j in range(len(form_cols)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.2 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax3, label='Korrelation')
plt.tight_layout()
plt.savefig("../data/form_streaks_heatmap.png", dpi=150, bbox_inches='tight')
print("✅ Chart 3 gespeichert: data/form_streaks_heatmap.png")

# ========================================================
# CHART 4: Das Alters-Paradoxon
# ========================================================
try:
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Gruppiere nach gerundetem Altersunterschied und berechne die Siegwahrscheinlichkeit
    df_age = df.copy()
    df_age['diff_age_rounded'] = df_age['diff_age'].round()
    age_prob = df_age.groupby('diff_age_rounded')['target'].agg(['mean', 'count']).reset_index()
    
    # Filtere extreme Ausreißer (z.B. Altersunterschiede mit weniger als 15 Kämpfen)
    age_prob = age_prob[age_prob['count'] >= 15]
    
    # Bubble Chart (Größe der Blase = Anzahl der Kämpfe)
    scatter = ax4.scatter(age_prob['diff_age_rounded'], age_prob['mean']*100, 
                          color='#3498db', s=age_prob['count']*0.8, alpha=0.7, 
                          edgecolor='white', linewidth=0.5,
                          label='Win % bei exakt X Jahren Unterschied (Größe = Sample Size)')
    
    # Polynomische Trendlinie (Grad 3 für glattere Kurve statt harter gerader Linie)
    z = np.polyfit(age_prob['diff_age_rounded'], age_prob['mean']*100, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(age_prob['diff_age_rounded'].min(), age_prob['diff_age_rounded'].max(), 100)
    ax4.plot(x_smooth, p(x_smooth), color='#e74c3c', linewidth=2.5, label='Trend (Polynomisch)')
    
    ax4.axvline(x=0, color='#2c3e50', linewidth=1.5, linestyle='-')
    ax4.axhline(y=50, color='gray', linewidth=1, linestyle='--', alpha=0.8)
    
    ax4.set_xlabel('Altersunterschied in Jahren (Positiv = Roter Kämpfer ist älter)', fontsize=12)
    ax4.set_ylabel('Siegwahrscheinlichkeit (Rote Ecke) in %', fontsize=12)
    ax4.set_title('Das Alters-Paradoxon: MMA ist "A Young Man\'s Game"', fontsize=14, fontweight='bold')
    
    # Text-Annotationen zur Erklärung
    ax4.text(-8, 65, 'Rot ist jünger\nhöhere Gewinnchance!', fontsize=10, color='#2ecc71', fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#2ecc71'))
    ax4.text(8, 35, 'Rot ist älter\ngeringere Gewinnchance!', fontsize=10, color='#e74c3c', fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c'))
    
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../data/alters_paradoxon.png", dpi=150, bbox_inches='tight')
    print("✅ Chart 4 gespeichert: data/alters_paradoxon.png")
except Exception as e:
    print(f"❌ Fehler bei Chart 4: {e}")

print("\n📊 Alle 4 Visualisierungen gespeichert in /data/!")
