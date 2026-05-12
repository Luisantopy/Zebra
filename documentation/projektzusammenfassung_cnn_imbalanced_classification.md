# Projektzusammenfassung – CNN für unausgeglichene Bildklassifikation

# 1. Ausgangslage und Ziel

Im Projekt wurde ein CNN zur binären Bildklassifikation aufgebaut.
Die Klassenverteilung war dabei stark unausgewogen:

- Klasse `n`: ~15’000 Samples
- Klasse `y`: ~270 Samples

Dadurch entstand sofort das zentrale Problem:

> Ein Modell kann sehr hohe Accuracy erreichen, obwohl es die positive Klasse fast nie erkennt.

Die ersten Baseline-Experimente bestätigten genau dieses Verhalten:

- Accuracy ~98%
- Recall = 0
- F1 = 0

---

# 2. Aufbau der Modellarchitektur

Es wurde ein relativ kleines CNN verwendet:

- 3 Conv-Blöcke
- jeweils:
  - Conv2D
  - ReLU
  - MaxPooling
- danach:
  - AdaptiveAvgPooling
  - kleiner Fully Connected Head

Die Architektur wurde bewusst einfach gehalten, um:

- die Trainingsdynamik besser analysieren zu können
- Overfitting bei kleinem positiven Datensatz zu vermeiden
- Hyperparameter systematisch untersuchen zu können

Wichtige Erkenntnis:

- Die Architektur selbst war nicht der Hauptengpass.
- Die Trainingsdynamik und Datenbalance hatten deutlich stärkeren Einfluss.

---

# 3. Umgang mit dem Klassenungleichgewicht

## Weighted Sampler (`alpha`)

Da die positive Klasse extrem selten war, wurde ein Weighted Sampler eingeführt.
Dabei steuerte `alpha`, wie stark positive Samples bevorzugt werden.

Beobachtungen:

| alpha | Verhalten |
|---|---|
| 0.3 | Modell ignoriert positive Klasse |
| 0.5 | leichte Verbesserung |
| 0.9 | hoher Recall, aber schlechte Precision |

Wichtige Erkenntnis:

> Der Sampler beeinflusst direkt den Recall/Precision-Tradeoff.

---

# 4. Einführung von Threshold Tuning

Anfangs wurde der Standard-Threshold von `0.5` verwendet.

Das führte oft dazu:

- positive Klasse kaum erkannt
- Recall extrem niedrig

Daraufhin wurde ein eigenes Threshold-Tuning implementiert:

- Wahrscheinlichkeiten über Softmax
- Threshold-Grid-Search
- Auswahl nach:
  - Mindest-Recall
  - maximalem F1

Dadurch verbesserten sich die Ergebnisse deutlich.

Beispiel:

- ohne Threshold-Tuning:
  - F1 ≈ 0.25
- mit optimiertem Threshold:
  - F1 ≈ 0.35

Wichtige Erkenntnis:

> Der Threshold war ein wichtiger Hebel im Projekt.

---

# 5. Adam vs SGD

Es wurden verschiedene Optimizer getestet:

## Adam

Vorteile:

- lernt schnell
- stabil bei kleinen Learning Rates

Nachteile:

- starkes Overfitting
- teilweise schlechte Generalisierung

Beispiel:

- Training nahezu perfekt
- Validation/Test deutlich schlechter

---

## SGD

Anfangs deutlich instabiler, aber:

- bessere Generalisierung möglich
- deutlich höhere Peak-F1-Werte erreichbar

Wichtige Erkenntnis:

> SGD konnte bessere Modelle finden, war aber extrem sensitiv auf Hyperparameter und Seeds.

---

# 6. Learning Rate als kritischster Hyperparameter

Es zeigte sich sehr schnell:

> Kleine Änderungen der Learning Rate führten zu komplett anderem Verhalten.

Getestet wurden u.a.:

- 0.05
- 0.055
- 0.06
- 0.061
- 0.065
- 0.071
- 0.075
- 0.1

Die Resultate schwankten massiv.

Beispiele:

## lr = 0.05

- einzelne Seeds:
  - F1 ≈ 0.56
- andere:
  - F1 ≈ 0.10

---

## lr = 0.075

- deutlich schlechter
- oft Kollaps

---

## lr = 0.06

Teilweise gute Mittelwerte:

- Avg F1 ≈ 0.45
- aber hohe Varianz

---

## lr = 0.061

Einige sehr starke Runs:

- F1 bis ~0.79

Aber weiterhin instabil:

- andere Seeds nur ~0.13

---

# 7. Seed-Instabilität entdeckt

Ein interessante Erkenntnis:

> Einzelne gute Runs waren nicht zuverlässig reproduzierbar.

Deshalb wurden identische Konfigurationen mit mehreren Seeds getestet.

Ergebnis:

- gleiche Hyperparameter
- komplett unterschiedliche Ergebnisse

Beispiel:

- F1:
  - 0.06
  - 0.50
  - 0.80

Wichtige Erkenntnis:

> Das Modell war nicht stabil genug für zuverlässige Generalisierung.

Deshalb wurde später nicht mehr nur die beste Einzelperformance betrachtet, sondern:

- Mittelwert über Seeds
- Standardabweichung
- Minimum / Maximum

---

# 8. Optuna für Hyperparameter-Tuning

Um systematischer zu suchen, wurde Optuna eingeführt.

Getunt wurden u.a.:

- learning rate
- alpha
- momentum
- min_recall

Später wurde zusätzlich:

- Mittelwert über mehrere Seeds
- minus Standardabweichung

optimiert, um Instabilität zu bestrafen.

---

## Beste gefundene Konfigurationen

Beispiele:

```python
lr ≈ 0.071
alpha ≈ 0.60
momentum ≈ 0.87
```

Einzelne Trials erreichten:

- Val F1 ≈ 0.75
- einzelne Test-F1s bis ~0.80

Aber:

- weiterhin starke Seed-Abhängigkeit

---

# 9. Regularisierung & Stabilisierung

Es wurden getestet:

## Dropout

- grösserer Head
- Dropout 0.2

Ergebnis:

- keine echte Stabilisierung
- häufig schlechtere Precision/F1

---

## L2-Regularisierung (`weight_decay`)

Ergebnis:

- stabilisierte leicht
- reduzierte aber Peak-Performance

Wichtige Erkenntnis:

> Das Hauptproblem war nicht klassisches Overfitting, sondern Trainingsinstabilität.

---

# 10. Data Augmentation

Augmentation rückte mehrfach im Projekt in den Fokus.

## Asymmetrische Augmentation

Wichtige Idee:

- positive Klasse (`y`) stärker augmentieren
- negative Klasse (`n`) nur leicht augmentieren

Getestet wurden u.a.:

- HorizontalFlip
- VerticalFlip
- Rotation
- Perspective
- ColorJitter
- GaussianBlur

---

## Beobachtungen

### Zu aggressive Augmentation

führte oft zu:

- schlechterer Precision
- instabilem Training
- Kollaps

---

### Moderate asymmetrische Augmentation

brachte leichte Verbesserungen:

- F1 ~0.11 statt ~0.09

Aber:

> Augmentation allein löste die Instabilität nicht.

---

# 11. Aktueller Stand

Die wichtigsten Erkenntnisse bisher:

## Was funktioniert hat

✅ Threshold Tuning  
✅ Weighted Sampling  
✅ SGD statt Adam  
✅ mittlere Learning Rates (~0.055–0.065)  
✅ mehrere Seeds evaluieren  
✅ Optuna für Hyperparameter-Suche  

---

## Was problematisch blieb

❌ starke Seed-Abhängigkeit  
❌ einzelne Trainings kollabieren komplett  
❌ grosse Schwankungen zwischen Runs  
❌ Precision/Recall schwer gleichzeitig stabil hoch zu halten  

---

# 12. Wichtigste Gesamt-Erkenntnis

Das Projekt entwickelte sich von:

> „Wie erreiche ich hohe Accuracy?“

zu:

> „Wie bekomme ich ein stabiles Modell auf einem extrem unausgeglichenen Datensatz?“

Die zentrale Herausforderung war letztlich nicht die Architektur selbst, sondern:

- Klassenungleichgewicht
- Thresholding
- Optimierungsdynamik
- Instabilität zwischen Seeds
- empfindliche Hyperparameter-Abhängigkeiten

Das Projekt hat deshalb einen starken Fokus auf:

- Reproduzierbarkeit
- Stabilität
- Hyperparameter-Tuning
- robuste Evaluation

entwickelt.