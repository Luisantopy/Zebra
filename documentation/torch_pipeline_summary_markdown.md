# Zusammenfassung der aktuellen Torch-Pipeline

## Überblick

Die aktuelle Pipeline lädt Trainings-, Validierungs- und Testdaten, trainiert ein CNN-Modell mit CrossEntropyLoss, nutzt Weighted Sampling gegen Klassenungleichgewicht, speichert das beste Modell anhand der Validation-F1 und führt anschließend Threshold-Tuning auf dem Validation-Set durch. Danach erfolgt die finale Bewertung auf dem Test-Set inklusive Probability Histogram, Precision-Recall-Kurve und Confusion Matrix.

Die wichtigsten Konfigurationsparameter sind aktuell:

```python
lr
alpha
momentum
min_recall
y_aug_params
```

---

# Datenverarbeitung

## Conditional Data Augmentation

Die Daten werden über `ImageFolder` geladen. Zusätzlich wurde eine eigene Klasse `ConditionalImageFolder` implementiert, damit Bilder der Klasse `y` anders augmentiert werden können als Bilder der Klasse `n`.

Ziel:
- stärkere Augmentation für die seltene Positivklasse
- höhere Varianz der Minderheitsklasse
- bessere Generalisierung

## Augmentation für Klasse `n`

Die Negativklasse wird nur leicht augmentiert:

```python
RandomHorizontalFlip(p=0.1)
ColorJitter(0.1, 0.1, 0.1)
```

Nutzen:
- leichte Robustheit gegen kleine Bildvariationen
- verhindert unnötige Verfremdung der Mehrheitklasse

## Augmentation für Klasse `y`

Die Positivklasse wird deutlich stärker augmentiert:

```python
RandomHorizontalFlip
RandomVerticalFlip
ColorJitter
RandomPerspective
RandomRotation
```

Aktuelle Default-Werte:

```python
hflip_p=0.5
vflip_p=0.2
brightness=0.3
contrast=0.5
saturation=0.5
perspective=0.2
rotation_deg=20
```

Nutzen:
- künstliche Vergrößerung der positiven Klasse
- bessere Robustheit gegenüber Perspektive, Licht und Orientierung
- geringere Gefahr von Overfitting auf wenige Zebra-Crossing-Muster

## Validation und Test

Validation- und Testdaten werden nicht augmentiert.

Es erfolgt nur:

```python
ToTensor
Normalize
```

Nutzen:
- realistische Evaluation
- vergleichbare Wahrscheinlichkeitsverteilungen
- keine künstliche Verzerrung der Metriken

---

# Sampling und Klassenungleichgewicht

## WeightedRandomSampler

Da die Daten stark unausgeglichen sind, wird ein `WeightedRandomSampler` verwendet.

Der Parameter:

```python
alpha
```

steuert, wie stark seltene Klassen bevorzugt werden.

Die aktuelle Version nutzt zusätzlich einen deterministischen Generator mit Seed.

Nutzen:
- mehr positive Beispiele pro Epoche
- höhere Wahrscheinlichkeit, Recall zu lernen
- verhindert Kollaps zur Negativklasse
- bessere Reproduzierbarkeit

## Getestete Alternativen

### BCE mit `pos_weight`

Wurde getestet, war aber nicht stabil besser als CrossEntropy.

### Zusätzliche Class Weights

Führten teilweise zu:
- instabilem Verhalten
- aggressivem Positive-Bias
- vielen False Positives

### Kein Sampling

Ohne Sampling lernte das Modell oft fast ausschließlich die Negativklasse.

---

# Modellarchitektur

## Aktuelles Modell: `CNNCrossEntropy`

Die aktuelle Architektur besteht aus drei Conv-Blöcken:

```text
Conv2d(3 → 32) + ReLU + MaxPool
Conv2d(32 → 64) + ReLU + MaxPool
Conv2d(64 → 128) + ReLU + MaxPool
AdaptiveAvgPool2d(1,1)
Flatten
Linear(128 → 64) + ReLU
Linear(64 → 2)
```

Output:

```text
[batch_size, 2]
```

Es wird bewusst keine Softmax im Modell verwendet, da `CrossEntropyLoss` direkt mit Logits arbeitet.

## Nutzen der Architektur

- einfache und transparente Struktur
- geringe Komplexität
- reproduzierbare Trainingsdynamik
- leichter interpretierbar als große pretrained Modelle
- geeignet für schnelle Experimente

## AdaptiveAvgPool

Statt großer Fully Connected Layer wird:

```python
AdaptiveAvgPool2d((1,1))
```

verwendet.

Nutzen:
- starke Reduktion der Parameteranzahl
- weniger Overfitting
- stabilere Trainingsdynamik

---

# Getestete Architektur-Alternativen

## BCE-Modell mit einem Output

Wurde getestet:

```python
CNNBinary
```

mit:

```python
BCEWithLogitsLoss
```

Ergebnis:
- keine stabile Verbesserung
- teilweise schlechtere Trennung

## Größerer Head

Wurde getestet.

Ergebnis:
- kaum Verbesserung
- teilweise stärkere Instabilität

## Dropout

Dropout wurde getestet, ist aktuell aber deaktiviert.

Grund:
- erhöhte teilweise die Instabilität
- keine konsistente Verbesserung

## BatchNorm

BatchNorm ist vorbereitet, aktuell aber auskommentiert.

Grund:
- zunächst Fokus auf reproduzierbare Baseline
- zusätzliche Komponenten erschweren Fehlersuche

## Keras-CNN

Eine äquivalente Keras-Version wurde implementiert.

Ergebnis:
- keine grundsätzlich bessere Stabilität
- ähnliche Seed-Abhängigkeit

## EfficientNet

EfficientNetB0 mit Pretraining wurde getestet.

Ergebnis:
- nicht automatisch besser
- Fine-Tuning deutlich komplexer
- sehr sensitiv auf LR und Unfreezing-Strategie

Erkenntnis:
- größere pretrained Modelle lösen das Problem nicht automatisch

---

# Loss und Optimizer

## Aktuelle Loss

```python
CrossEntropyLoss
```

Nutzen:
- stabile Optimierung
- direkte Unterstützung für Multi-Class Logits
- gutes Zusammenspiel mit Threshold-Tuning

## Optimizer

Aktuell:

```python
SGD + Momentum
```

Wichtige Parameter:

```python
lr
momentum
```

## Erkenntnisse aus den Experimenten

Die Learning Rate ist aktuell der wichtigste Hebel.

Besonders gute Ergebnisse wurden beobachtet bei:

```python
lr ≈ 0.05
```

Niedrige oder ungeeignete LR führten oft dazu, dass das Modell fast ausschließlich `n` vorhersagte.

## Getestete Alternativen

### Adam

Wurde getestet.

Ergebnis:
- teilweise schnelle Konvergenz
- aber keine stabil besseren Ergebnisse

### Weight Decay / L2-Regularisierung

Wurde getestet.

Ergebnis:
- teilweise stärkere Instabilität
- keine klare Verbesserung

### ReduceLROnPlateau

Ist implementiert, aktuell aber deaktiviert.

Grund:
- zunächst stabile Baseline reproduzieren
- automatische LR-Änderungen erschweren Debugging

---

# Training

## Early Stopping

Das beste Modell wird anhand der Validation-F1 gespeichert.

Aktuelle Konfiguration:

```python
patience=13
mode="max"
```

Nutzen:
- speichert bestes Modell statt letzter Epoche
- wichtig bei stark schwankenden Validation-Metriken
- reduziert Risiko schlechter finaler Checkpoints

## Logging

Pro Epoche werden gespeichert:

```text
Train Loss
Train Accuracy
Train Recall
Train Precision
Train F1
Validation Loss
Validation Accuracy
Validation Recall
Validation Precision
Validation F1
```

Nutzen:
- Analyse von Overfitting
- Vergleich verschiedener LR- und Sampler-Konfigurationen
- Reproduzierbarkeit früherer Runs

---

# Threshold-Tuning

Nach dem Training wird auf dem Validation-Set ein optimaler Threshold gesucht.

Auswahlkriterium:

```python
metric="f1"
```

unter Nebenbedingung:

```python
min_recall
```

Nutzen:
- bessere Kontrolle über Recall/Precision-Tradeoff
- sinnvoll bei stark unausgeglichenen Daten
- deutlich informativer als Accuracy alleine

---

# Zusätzliche Evaluationsmethoden

## Probability Histogram

Visualisiert die Wahrscheinlichkeitsverteilungen von:
- Klasse `n`
- Klasse `y`

Nutzen:
- Analyse der Trennbarkeit
- Erkennung von Bias zur Negativklasse
- Identifikation schlechter Kalibrierung

## Precision-Recall-Kurve

Zeigt den Tradeoff zwischen:
- Precision
- Recall

Nutzen:
- bessere Bewertung bei Klassenimbalance
- aussagekräftiger als ROC-Curve in diesem Szenario

## Confusion Matrix

Zeigt:

```text
True Positives
False Positives
True Negatives
False Negatives
```

Nutzen:
- direkte Interpretation der Fehlerarten
- wichtig für reale Zebra-Crossing-Erkennung
- zeigt, ob Recall oder Precision das Hauptproblem ist

---

# Aktueller Erkenntnisstand

Die wichtigsten Erkenntnisse bisher:

1. Die Architektur selbst ist grundsätzlich ausreichend.
2. Die Trainingsdynamik ist wichtiger als Modellgröße.
3. Learning Rate ist aktuell der wichtigste Hyperparameter.
4. Sampling beeinflusst Recall massiv.
5. Zu aggressive Regularisierung verschlechterte häufig die Stabilität.
6. Große pretrained Modelle waren nicht automatisch besser.
7. Gute Runs sind reproduzierbar, wenn:
   - LR
   - Seed
   - Sampling
   - Threshold
   konsistent gehalten werden.

---

# Geplante nächste Schritte

Die aktuell sinnvolle Reihenfolge für weitere Experimente:

```text
1. Learning Rate fein optimieren
2. alpha (Weighted Sampling) optimieren
3. momentum optimieren
4. erst danach Data Augmentation optimieren
```

Wichtige Erkenntnis:

Nicht zu viele Komponenten gleichzeitig verändern, da sonst nicht nachvollziehbar ist, welche Änderung welchen Effekt verursacht.

