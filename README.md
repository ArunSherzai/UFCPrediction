# UFC Fight Predictor

Web-App, die den Ausgang von UFC-Kämpfen vorhersagt: wer gewinnt, in welcher Runde der Kampf endet und wie (KO/TKO, Aufgabe, Punktentscheidung). Zu jeder Vorhersage zeigt die App, welche Faktoren den Ausschlag geben, und vergleicht sie mit den Wettquoten der Buchmacher.

Entwickelt von **David Vukasovic** und **Arun Sherzai**, großteils im Pair Programming an einem Rechner.

## Features

- **Vorhersage** von Sieger, Runde und Siegmethode für zwei frei wählbare Kämpfer
- **„Tale of the Tape“** – Kämpfervergleich mit Reichweite, Größe, Alter, Schlag- und Takedown-Statistiken
- **Erklärung**, warum das Modell so entscheidet
- **Vergleich mit Wettquoten**
- **Datenpipeline**, die neue UFC-Events automatisch nachlädt und alle Modelle neu trainiert

## Daten & Modelle

- 8.427 Kämpfe mit 134 Merkmalen, Stand März 2026, Quelle: ufcstats.com
- Merkmale als Differenz zwischen beiden Kämpfern (Alter, Reichweite, Kontrollzeit, Knockdowns …)
- Trainings- und Testdaten **zeitlich getrennt**: getestet wird nur auf Kämpfen, die nach den Trainingsdaten stattfanden

| Modell | Accuracy | Vergleich |
| --- | --- | --- |
| Sieger (Logistic Regression) | 60,5 % | 56,1 % wenn man immer die häufigere Ecke tippt |
| Sieger (Random Forest) | 61,9 % | ausgeglichene Accuracy niedriger, daher Logistic Regression als Standard |
| Runde | 45,2 % | |
| Siegmethode | 43,8 % | |

Wichtigster Einzelfaktor ist der Altersunterschied: Der jüngere Kämpfer gewinnt deutlich häufiger.

## Tech-Stack

Python · pandas · NumPy · scikit-learn · Streamlit · BeautifulSoup · Matplotlib · Seaborn

## Starten

```bash
pip install pandas numpy scikit-learn streamlit requests beautifulsoup4 matplotlib seaborn
cd script
streamlit run app.py
```

Daten aktualisieren und Modelle neu trainieren:

```bash
python refresh_current_data.py
```

Für Wettquoten über The Odds API optional die Umgebungsvariable `THE_ODDS_API_KEY` setzen.

**Kämpferfotos** sind aus Lizenzgründen nicht im Repository. Mit `python download_fighter_images.py` lassen sie sich lokal herunterladen, ohne Fotos zeigt die App einen Platzhalter.

## Aufteilung

Daten, Modelle und App gemeinsam im Pair Programming. Davon zusätzlich:

- **David:** Datenbereinigung, Feature Engineering, Modelle und Datenpipeline
- **Arun:** Oberfläche des Kämpfervergleichs („Tale of the Tape“) und Datenvisualisierung
