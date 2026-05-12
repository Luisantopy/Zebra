Zusammenfassung des bisherigen Projekts

---

# 🧠 1. Baseline-Modell aufgesetzt

* Einfaches CNN mit:

  * Conv → ReLU → Pool (3 Blöcke)
  * Linear Head
* Loss: `CrossEntropyLoss`
* Standard-Threshold = 0.5

**Warum:**

* **Baseline** nötig, um Verbesserungen messen zu können
* CrossEntropy ist Standard für Klassifikation ––> stabil & gut getestet

👉 Ergebnis:

* Sehr hohe Accuracy (~0.97)
* Aber extrem schlechter Recall / F1 (oft 0.0)

➡️ Erste wichtige Erkenntnis:

> Accuracy ist hier irreführend (stark unausgewogene Klassen)

---

# ⚖️ 2. Umgang mit Klassen-Ungleichgewicht

* Weighted Sampler (`alpha`-Parameter)

**Warum:**

* Datensatz ist stark **imbalanced**
* Modell lernt sonst: “immer negative Klasse vorhersagen”

👉 Effekt:

* Recall steigt deutlich
* Precision sinkt stark

➡️ Erkenntnis:

> Trade-off zwischen Recall und Precision wird zentral

---

# 🎯 3. Threshold Tuning eingeführt

* Softmax → Wahrscheinlichkeiten
* Schwellenwert variabel gemacht
* Grid Search über Thresholds
* Auswahl mit:

  * Mindest-Recall (`min_recall`)
  * bestes F1

**Warum:**

* Default Threshold = 0.5 ist selten optimal bei unbalanciertem Datensatz
* Ziel:

  * Recall kontrollieren
  * Precision nicht komplett zerstören

👉 Effekt:

* Massive Verbesserung gegenüber fixem Threshold

➡️ Erkenntnis:

> Threshold ist ein entscheidender Hebel

---

# 🔁 4. Verschiedene Optimizer & Learning Rates getestet

* Adam vs SGD
* verschiedene Learning Rates (1e-4 → 1e-1)

**Warum:**

* Optimizer bestimmt:

  * Konvergenzverhalten
  * Stabilität
  * Generalisierung

👉 Ergebnis:

* Adam: schnell, aber teilweise overfitting / instabil
* SGD: schwieriger zu tunen, aber bessere Ergebnisse möglich

➡️ wichtige Erkenntnis:

> SGD + richtige LR → deutlich bessere Performance als Adam

---

# 📉 5. Learning Rate als kritischer Faktor erkannt

* kleine Änderungen in LR → komplett andere Ergebnisse

**Warum das wichtig ist:**

* LR bestimmt:

  * ob Modell konvergiert
  * ob es divergiert
  * ob es lokale Minima findet

👉 Beispiele:

* zu klein → lernt kaum
* zu gross → kollabiert (wie in manchen runs)

➡️ Erkenntnis:

> LR ist ein wichtiger Hyperparameter im Setup

---

# 🔍 6. Instabilität entdeckt

* gleiche Experimente mehrfach mit verschiedenen Seeds laufen lassen

**Warum:**

* Training enthält Zufall:

  * Initialisierung
  * Sampling
  * Augmentation

👉 Ergebnis:

```
Test F1: 0.06 → 0.50
```

➡️ extrem grosse Streuung

> Setup ist nicht stabil

---

# 🤖 7. Optuna für Hyperparameter-Suche eingeführt

* automatische Suche für:

  * learning rate
  * sampler alpha
  * momentum
  * min_recall

**Warum:**

* manuelles Tuning ineffizient
* viele Parameter interagieren miteinander

👉 Ergebnis:

* deutlich bessere Konfiguration gefunden
* z. B.:

  * SGD
  * LR ~0.05
  * sinnvoller alpha + momentum

➡️ Erkenntnis:

> automatisches Tuning ist hilfreicher als manuelles Tuning

---

# 📊 8. Multi-Seed Evaluation integriert

* bestes Setup über mehrere Seeds getestet
* Mittelwert + Standardabweichung berechnet

**Warum:**

* einzelner Run ist nicht zuverlässig
* Ziel: robuste Modelle

👉 Ergebnis:

* grosse Varianz sichtbar geworden

➡️ Erkenntnis:

> gutes Modell = gute Performance + geringe Varianz

---

# 🧪 9. Objective verbessert (Robustheit berücksichtigen)

* nicht nur Mean optimieren
* sondern auch Stabilität

**Warum:**

* verhindert Zufallstreffer
* bevorzugt robuste Lösungen

---

# 🧱 10. Architektur überprüft

* überprüft:

  * Output Layer Aktivierung
  * Struktur des CNN

**Warum:**

* sicherstellen, dass:

  * Loss korrekt verwendet wird
  * keine falschen Aktivierungen drin sind

👉 Erkenntnis:

* CrossEntropy nutzt intern Softmax → alles korrekt
* Grösseres Modell würde nicht unbedingt die Stabilität erhöhen (Train F1 ist oft sehr hoch, das Modell kann offensichtlich gut lernen; Problem ist Stabilität und Generalisierungsfähigkeit)

---

# 💾 11. Trainingspipeline strukturiert

* `helpers.py`
* `training_pipeline.py`
* Trennung von:

  * Training
  * Evaluation
  * Threshold tuning

**Warum:**

* bessere Wartbarkeit
* einfachere Experimente

---

# 🧩 Gesamtbild

1. Baseline
2. Problem erkannt (Imbalance)
3. Sampling / Loss angepasst
4. Threshold eingeführt
5. Optimizer & LR untersucht
6. Instabilität entdeckt
7. Automatisches Tuning (Optuna)
8. Robustheit bewertet (Multi-seed)

---

# 🧠 Wichtigste Learnings

### 1. Accuracy ist nutzlos bei Imbalance

→ F1 / Recall entscheidend

### 2. Threshold ist extrem wichtig

→ oft wichtiger als Modelländerungen

### 3. Learning Rate ist kritisch

→ bestimmt alles

### 4. Einzelne Runs sind wertlos

→ immer mehrere Seeds testen

### 5. Bestes Modell ≠ höchster Score

→ Stabilität zählt

---
