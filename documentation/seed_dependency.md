# Seed-Abhängigkeit und Reproduzierbarkeit bei Deep Learning / Image Classification

## Beobachtung im Projekt

Im aktuellen Zebra-Crossing-Projekt wurde festgestellt, dass identische Trainingsläufe trotz gleicher:

- Daten
- Hyperparameter
- Modellarchitektur
- Seeds

teilweise deutlich unterschiedliche Ergebnisse liefern.

Besonders betroffen waren:
- F1-Score
- Recall
- Threshold-Selektion
- Training Stability

Dies trat sowohl bei:
- PyTorch-CNNs
- EfficientNet
- WeightedRandomSampler
- starker Klassenimbalance

auf.

---

# Warum passiert das?

Seed-Abhängigkeit ist ein bekanntes Problem im Deep Learning.

Selbst wenn:

```python
set_seed(42)

gesetzt wird, bedeutet das nicht automatisch vollständig reproduzierbare Ergebnisse.

Viele Komponenten im Training enthalten zusätzliche Zufallsquellen oder nichtdeterministische Operationen.

Typische Ursachen
1. Zufällige Gewichtinitialisierung

Neuronale Netze starten mit zufälligen Gewichten.

Kleine Unterschiede zu Beginn können später große Unterschiede in:

Decision Boundaries
Thresholds
Generalisierung

verursachen.

2. DataLoader / Batch-Reihenfolge

Die Reihenfolge der Trainingsbilder beeinflusst SGD stark.

Besonders relevant bei:

kleinen Datensätzen
Klassenimbalance
Weighted Sampling

Im Projekt war dies wahrscheinlich einer der Hauptgründe für die Unterschiede zwischen alten und neuen Runs.

3. WeightedRandomSampler

Der WeightedRandomSampler erzeugt zufällige Batch-Zusammensetzungen.

Wenn kein fixer Generator verwendet wird:

generator=torch.Generator().manual_seed(seed)

sind Runs trotz globalem Seed nicht reproduzierbar.

4. Data Augmentation

Augmentation erzeugt zusätzliche Zufallsquellen:

Rotation
Flip
Contrast
Perspective
Brightness

Bereits kleine Unterschiede können:

andere Features
andere lokale Minima
andere Thresholds

verursachen.

5. GPU / CUDA / MPS Nichtdeterminismus

Viele GPU-Operationen sind nicht vollständig deterministisch.

PyTorch dokumentiert explizit:

Vollständige Reproduzierbarkeit zwischen Geräten, Plattformen und Versionen wird nicht garantiert.

Besonders betroffen:

CUDA
Apple MPS
parallele DataLoader
6. Floating Point Effekte

Floating-Point-Arithmetik ist nicht assoziativ:

(a + b) + c != a + (b + c)

Dadurch können minimale numerische Unterschiede entstehen, die sich über viele Epochen verstärken.

Warum ist das Projekt besonders seed-sensitiv?

Das aktuelle Setup enthält mehrere Faktoren, die Instabilität verstärken:

Faktor	Effekt
starke Klassenimbalance	hohe Threshold-Sensitivität
WeightedRandomSampler	zufällige Batch-Zusammensetzung
kleine positive Klasse	hohe Varianz
aggressive Augmentation	zusätzliche Zufallsquellen
EarlyStopping	andere Stopping-Epochen
Threshold-Tuning	verstärkt kleine Unterschiede
kleines CNN	weniger robuste Features


Wichtige Erkenntnis

Einzelne Seeds können stark schwanken.

Daher gelten heute in vielen Deep-Learning-Arbeiten:

Mittelwerte über mehrere Seeds
Standardabweichungen
robuste Evaluation

als deutlich aussagekräftiger als einzelne „Best Runs“.

Fazit

Die beobachtete Seed-Abhängigkeit ist kein ungewöhnlicher Fehler, sondern ein bekanntes Problem moderner Deep-Learning-Systeme.

Besonders kleine und unausgewogene Datensätze reagieren stark auf:

Sampling
Initialisierung
Augmentation
Threshold-Tuning
Learning Rate

Die wichtigste praktische Verbesserung im Projekt war:

deterministischer WeightedRandomSampler
niedrigere Learning Rate
systematische Evaluation über mehrere Seeds