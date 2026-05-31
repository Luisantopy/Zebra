# Bericht: Entwicklung einer Deep-Learning-Pipeline zur Erkennung von Zebrastreifen in Luftbildern

## Projektziel

Ziel des Projekts war die Entwicklung eines robusten 
Deep-Learning-Modells zur binären Klassifikation von Luftbildern:

* Klasse `y`: Bild enthält Zebrastreifen
* Klasse `n`: Bild enthält keinen Zebrastreifen

Der Fokus lag dabei auf:

* hoher Precision
* gleichzeitig brauchbarem Recall
* robuster Generalisierung
* reproduzierbaren Ergebnissen
* Analyse von Fehlerbildern (False Positives / False Negatives)
* Verständnis von Seed-Abhängigkeit und Training-Stabilität

Im Verlauf des Projekts wurden verschiedene Ansätze getestet:

* einfache CNNs
* grössere Torch-CNNs
* Keras-Modelle
* pretrained ResNet18
* verschiedene Sampling-Strategien
* Threshold-Tuning
* Data Augmentation
* Fine-Tuning
* Hyperparameteroptimierung mit Optuna

---

# 1. Datensatz und Problemcharakteristik

Der Datensatz besteht aus Luft- bzw. Satellitenbildern urbaner Szenen.

Positive Beispiele enthalten:

* Fussgängerstreifen
* teils verdeckte Zebrastreifen
* unterschiedliche Perspektiven
* Schatten
* komplexe Kreuzungen

Negative Beispiele enthalten:

* Strassen
* Parkplätze
* Vegetation
* Gebäude
* Kreuzungen ohne Zebrastreifen

---

# 2. Zentrale Herausforderungen des Datensatzes

Bereits früh zeigte sich:

## Starkes Klassenungleichgewicht

Die negative Klasse dominierte deutlich. Aufgrund des Klassenungleichgewichts 
wurden die Train, Test und Validierungsdatensätze *stratifiziert* erstellt. 

Durch die stratifizierte Aufteilung wurde sichergestellt, dass
die Klassenverteilung in Train-, Validation- und Testdaten
möglichst ähnlich bleibt.

Ohne Stratifizierung hätte insbesondere die kleine Positivklasse
zwischen den Splits stark schwanken können, was die Evaluation
zusätzlich instabil gemacht hätte.

Folgen:

* Modelle lernten schnell: ```"immer nein sagen" ```
* hohe Accuracy war leicht erreichbar
* Recall kollabierte häufig

## Schwierige Positivklasse

Positive Beispiele waren sehr heterogen:

* unterschiedliche Perspektiven
* teilweise kleine Zebraflächen
* Schatten
* Teilverdeckung durch Bäume
* ungewöhnliche Strassenlayouts

-----

# 3. Erste Modelle

Zu Beginn wurden einfache CNNs mit PyTorch getestet.
Ziel war es eine solide Basis zu finden und Schritt für Schritt zu 
optimieren, damit erkennbar ist, welche Veränderungen zielführend
sind.

## Torch CNN

Zunächst wurde ein kleines eigenes CNN verwendet.

### Architektur

Mehrere Conv-Blöcke:

```text
Conv2D
→ ReLU
→ MaxPool
```

anschliessend:

* AdaptiveAvgPool
* Flatten
* Fully Connected Head

## Varianten

### SimpleCNNCrossEntropy

* CrossEntropyLoss
* Ausgabe: ```[batch_size, num_classes] ```

### SimpleCNNBinary

* BCEWithLogitsLoss
* Ausgabe: ```[batch_size, 1] ```

---

# 4. Loss-Funktionen

## CrossEntropyLoss

Erwies sich insgesamt als stabiler.

Vorteile:

* sauberere Probability Separation
* stabileres Threshold-Tuning
* bessere Precision

## BCEWithLogitsLoss

Wurde getestet, aber letztlich nicht bevorzugt.

Probleme:

* instabilere Thresholds
* stärkere Sensitivität auf Class Imbalance
* schlechtere Probability Calibration

## Fazit zu Loss Funktionen

CrossEntropyLoss modelliert die beiden Klassen gemeinsam
über eine Softmax-Verteilung.

Dadurch entstehen häufig:
- besser separierte Wahrscheinlichkeiten
- stabilere Decision Boundaries
- robustere Thresholds

BCEWithLogitsLoss behandelt dagegen jede Klasse
unabhängig, was bei stark unausgeglichenen Klassen
zu instabileren Wahrscheinlichkeiten führen kann.

---

# 5. Sampling und Class Imbalance

Ein zentraler Teil des Projekts war der Umgang mit dem Klassenungleichgewicht.
Da davon auszugehen ist, dass ein realistischer Anwendungsfall ein ähnlich 
stark ausgeprägtes Unggleichgewicht erzeugen würde, wurde davon abgesehen, die 
Disbalance künstlich durch Weglassen von Negativ-Samples zu korrigieren.

## Ursprünglicher Ansatz

Class Weights in der Loss Function.

Probleme:

* teilweise instabiles Training
* Recall-Schwankungen
* stärkere Seed-Abhängigkeit

## WeightedRandomSampler

Der gewichtete Sampler erwies sich als zielführender:

```python
weight = 1 / class_count**alpha
```

mit:

* replacement=True
* reproduzierbarem Generator-Seed

Durch den WeightedRandomSampler wurden pro Epoche
mehr positive Beispiele gesehen, ohne dass der Datensatz
physisch verändert werden musste.

Dadurch konnte der Recall deutlich verbessert werden,
allerdings erhöhte aggressives Oversampling gleichzeitig
die Gefahr von Overfitting auf die Positivklasse.

### Alpha-Tuning

`alpha` kontrolliert die Stärke des Oversamplings.

Beobachtungen:

#### Zu kleines Alpha

→ Modell ignoriert Positivklasse

#### Zu grosses Alpha

→ Overfitting auf Positivklasse

### Typischer guter Bereich

`alpha ≈ 0.55–0.65`

## Data Augmentation

Die Augmentation wurde iterativ erweitert.

### Ursprüngliche Augmentations

* HorizontalFlip
* VerticalFlip
* ColorJitter
* Perspective
* Rotation

Von Augmentationen, welche das Bild so bearbeiten, dass unter Umständen 
kein Zebrastreifen mehr enthalten ist (zB. Crop) wurde abgesehen.

### Versionen

Es wurden manuelle Experimente mit verschiedenen Augmentationen getestet, 
sowohl mit stärkeren als auch mit schwächeren Parametern. Beispiel:

* stärkere Rotation
* stärkere Perspective
* RandomErasing

### Ergebnis

Zu aggressive Augmentation verschlechterte die Ergebnisse deutlich.

Folgen:

* mehr False Negatives
* unsicherere Positivklasse
* Verlust klarer Zebra-Muster

### Zentrale Erkenntnis

```Mehr Augmentation ≠ besser.```

Insbesondere:

* RandomErasing
* starke Perspective Distortion
* starke Rotation

verschlechterten die Robustheit.

### Erkenntnisse zu sinnvoller Augmentation

Am sinnvollsten erschienen:

* moderate Rotation
* leichte Perspective
* moderate ColorJitter

Nicht sinnvoll:

* aggressive Bildzerstörung
* extreme Geometrie
* starke künstliche Occlusion

---

# 6. Reproduzierbarkeit und Seed-Abhängigkeit

Ein zentrales Learning des Projekts war: ```Image Classification kann extrem seed-abhängig sein.```

### Beobachtungen

Mit identischer:

* Architektur
* Config
* Lernrate
* Augmentation

wurden teilweise massiv unterschiedliche Ergebnisse erzielt.

Beispiel:

```text
Avg Test F1: 0.4575 ± 0.2913
Min Test F1: 0.1518
Max Test F1: 0.8312
```

Die starke Seed-Abhängigkeit zeigte sich nicht nur
in den finalen Metriken, sondern bereits während des Trainings:

- unterschiedliche Probability Histograms
- unterschiedliche Precision-Recall-Kurven
- unterschiedliche Thresholds
- unterschiedliche False-Negative-Typen

Dadurch wurde klar, dass einzelne gute Runs
keine ausreichende Aussagekraft besitzen.

### Ursachen

Die Ursachen lagen vermutlich in:

* unterschiedlichen Batch-Reihenfolgen
* Oversampling-Dynamik
* lokalen Minima
* kleinen Positiv-Sets
* stochastischen Gradientenupdates

### Konsequenzen

Es wurde klar, dass ein einzner guter Run nicht bedeutet, dass das Modell stabil 
lernt. Dies wurde nicht nur an den Abweichungen über die Seeds deutlich sondern auch 
über die Probability-Histogramme und die Precision-Recall-Kurven. 
Um die Stabilität zu erhöhen, wurde:

* über mehrere Seeds evaluiert
* Mittelwert + Standardabweichung betrachtet

### Wichtige Erkenntnis

Einzelne Seeds führten zu guten Ergebnissen, während andere Seeds massiv
von diesen Ergebnissen abwichen.

Daher wurde die Objective Function angepasst:

```python
mean(F1) - std(F1)
```

Dadurch wurden instabile Konfigurationen bestraft.

Allein mit dem CNN Modell wurde allerdings auch nach diversen Experimenten 
(manuell und mit Optuna) keine Konfiguration gefunden, die keine starke
Seed-Abhängigkeit aufweist.

---

# 7. Hyperparameteroptimierung mit Optuna

Optuna wurde verwendet zur Optimierung von:

* Learning Rate
* Sampler Alpha
* teilweise Augmentation
* Threshold-Constraints


---

# 8. Learning Rate als Schlüsselfaktor

Eine der wichtigsten Erkenntnisse des Projekts:

```text
Learning Rate ist, vor allem bei Verwendung des SGD Optimizers, 
ein essentieller Hyperparameter. Die Learning Rate muss spezifisch 
an das verwendete Modell & den Optimizer angepasst werden, es gibt keine
"one-fits-all" Lösungen. 
```

## Beobachtungen bei der Optimierung der LR mit SGG

### Zu hohe LR
Führte zu:

* chaotischem Training
* Kollaps auf Negative Class
* Recall = 0
* instabilen Thresholds

Zu hohe Learning Rates führten häufig dazu,
dass das Modell sehr früh in schlechte lokale Minima
oder instabile Decision Boundaries konvergierte.

Dies zeigte sich insbesondere durch:
- Recall-Kollaps
- Probability Collapse
- starke Unterschiede zwischen Seeds

### Gute Bereiche

Die Lernrate wurde mir Optuna optimiert, als guter Bereich stellte sich 
eine Lernrate von ```lr ≈ 0.04–0.06``` heraus.

Besonders gute Ergebnisse: ```lr = 0.05```

---

# 9. Scheduler

ReduceLROnPlateau wurde getestet.

Ergebnis:

* deutlich schlechtere Generalisierung
* Probability Collapse
* Precision-Zusammenbruch

Der Scheduler reduzierte die Learning Rate häufig
zu Zeitpunkten, an denen die Validation-Metriken
stark schwankten.

Dadurch wurde das Modell teilweise in ungünstigen
Wahrscheinlichkeitsverteilungen „eingefroren“,
was zu schlechterer Probability Calibration führte.

Der Scheduler wurde deshalb verworfen.

---

# 10. Weight Decay

Weight Decay wurde ebenfalls getestet.

Schon kleine Werte `1e-5` oder `5e-5` 
verschlechterten die Ergebnisse deutlich.

Daraus wurde geschlossen:

* Overfitting war nicht das primäre Problem
* zusätzliche Regularisierung schadete eher

Weight Decay wurde verworfen.

---

# 11. Threshold-Tuning

Ein zentraler Bestandteil der Pipeline.

### Motivation

Standardthreshold ```0.5``` war häufig ungeeignet.

### Frühe Modelle

Zeigten:

* starke Überlappung der Klassen
* viele Threshold Scores zwischen 0.05–0.30

Dadurch hatte Threshold-Tuning grossen Einfluss.

----

# 12. Probability Histogram und PR Curves

Diese Plots wurden zu zentralen Analysewerkzeugen.

## Probability Histogram

Half zu erkennen:

* ob Klassen separiert werden
* ob Scores kollabieren
* ob das Modell unsicher ist

Die Probability Histograms erwiesen sich als besonders hilfreich,
um zu verstehen, ob das Modell:
- die Klassen sauber trennt
- unsichere Vorhersagen produziert
- oder vollständig kollabiert.

Insbesondere konnte beobachtet werden,
dass gute Modelle:
- negative Beispiele nahe 0
- positive Beispiele nahe 1
platzierten.

## PR Curve

Zeigte:

* Ranking-Qualität
* Precision-Recall-Tradeoff
* Modellkalibrierung

---

# 13. Keras-Experimente zur Untersuchung der Seed-Abhängigkeit

Nachdem in den frühen PyTorch-Experimenten eine starke Seed-Abhängigkeit beobachtet wurde, 
wurden zusätzlich Keras-Modelle getestet.

Ziel war nicht primär eine bessere Accuracy, sondern die Untersuchung der Frage:

Ist die Instabilität möglicherweise framework-spezifisch?

### Motivation

In PyTorch zeigte sich bereits früh:

- identische Hyperparameter
- identische Architektur
- identische Datenaufteilung

führten teilweise zu massiv unterschiedlichen Ergebnissen.

Typische Symptome:

- einzelne sehr gute Runs
- einzelne katastrophale Runs
- Recall-Kollaps
- starke Unterschiede in Probability Calibration
- hohe Standardabweichungen zwischen Seeds

### Ziel der Keras-Tests

Die Keras-Experimente dienten dazu:

- die Trainingsdynamik zu vergleichen
- Unterschiede in Initialisierung und Optimierung zu prüfen
- die Reproduzierbarkeit zwischen Frameworks zu analysieren
- auszuschliessen, dass die Instabilität spezifisch durch PyTorch verursacht wird

### Erkenntnisse

Die Ergebnisse zeigten:

- auch unter Keras blieb die Seed-Abhängigkeit bestehen
- gute Runs schwankten weiterhin stark
- die Instabilität war daher kein reines Framework-Problem

Dadurch wurde klar:

- Die Ursache liegt primär im Datensatz,
- der Klassenimbalance,
- der kleinen positiven Klasse
- und der Optimierungsdynamik.

### Wichtiges Learning

Die Keras-Experimente bestätigten:

Seed-Abhängigkeit ist bei Image Classification
ein reales strukturelles Problem.

Insbesondere bei:

- kleinen Datensätzen
- stark unausgeglichenen Klassen
- seltenen Positivbeispielen
- aggressivem Oversampling
- stochastischer Augmentation

---

# 14. Wechsel zu Pretrained Models

### Motivation

Die eigenen CNNs:

* lernten brauchbare Features
* hatten aber begrenzte Generalisierung

Ein pretrained Modell sollte:

* robustere Features
* bessere Generalisierung
* stabileres Training

liefern.

Um zu untersuche, ob das Modell stabiler wird (das heisst weniger Seed-abhängig und 
sicherer ist), wurde ein Pretrained Modell integriert.

### Wahl des Modells 
Aufgrund der Kompatibilität mit der bestehenden Modell-Pipeline und der
einfachen Integration in PyTorch, wurde ein ResNet18 integriert.

Vorteile:
* stabiler
* einfach zu trainieren
* weniger hypersensitiv
* sehr robust bei kleinen Datensätzen
* sehr gute Baseline

## ResNet18 Architektur

Verwendet wurde:

```python
torchvision.models.resnet18
```

mit:

```python
ResNet18_Weights.IMAGENET1K_V1
```

### Anpassung

Der finale Fully Connected Layer wurde ersetzt:

```python
self.backbone.fc = nn.Linear(
    in_features,
    num_classes,
)
```

## Fine-Tuning Strategie

Es wurden Hilfsfunktionen implementiert:

* freeze_backbone()
* unfreeze_last_block()
* unfreeze_all()


### Phase 1

Nur FC Head trainieren.

### Phase 2

Zusätzlich ```layer4``` freigeben.

### Phase 3

Optional komplettes Modell freigeben.
Dies wurde als nicht zielführend verworfen.


## Verhalten des pretrained Modells

Das ResNet18 zeigte:

* deutlich bessere Probability Separation
* höhere Precision
* stabilere Thresholds
* bessere Generalisierung: die Seed-Abhängigkeit war nicht mehr
problematisch, das Modell kann sehr gut unterscheiden

Das pretrained ResNet18 profitierte davon,
dass bereits allgemeine visuelle Merkmale
wie:
- Kanten
- Linien
- Texturen
- Kontraste
- Straßenstrukturen

gelernt waren.

Dadurch musste das Modell nicht mehr
bei zufälliger Initialisierung beginnen,
was die Trainingsstabilität deutlich verbesserte.

Allerdings war das Modell anfangs eher konservativ bzgl. Positives. 
In den Konfusionsmatrizen wurde dies durch deutlich mehr FN als FP 
deutlich. In einer realen Anwendung wären FP allerdings deutlich 
weniger problematisch als FN. 
Bsp: 
Confusion matrix:
[[5742    6]
 [  79  173]]

Das Tuning des Modell konzentrierte sich als weniger auf eine
Stabilisierung oder Verbesserung der Generalisierungsfähigkeit 
(hier wurden bereits sehr gute Werte erreicht) sondern auf 
eine Verbesserung des Recalls. 

## Threshold-Tuning

Mit dem Pretrained Modell wurde Threshold-Tuning zunehmend unwichtiger.
Die Probability-Histogramme zeigen eine klare Probability Separation,
die Negatives sind nahe 0, die Ppositives nahe 1.

Beispiel:

```text
Threshold 0.30 → F1 0.8875
Threshold 0.60 → F1 0.8803
```

Das zeigte ```gute Kalibrierung und saubere Klassentrennung```.

---

# 18. Analyse von False Negatives

Ein sehr wichtiger Projektteil.

### Erkenntnis

Die False Negatives waren häufig:

* keine Edge Cases
* klar erkennbare Zebrastreifen

### Typische problematische Muster

* Schatten
* starke Perspektiven
* diagonale Strassen
* Teilverdeckung
* kleine Zebraflächen
* ungewöhnliche Strassenlayouts

### Wichtiges Learning

Das Problem war mit dem Pretrained Modell nicht mehr ```"Modell versteht Zebra nicht"```,
sondern ```"das Modell versteht bestimmte Muster nicht"```.

---

# 19. Hard Positive Mining

Nach der Analyse der False Negatives zeigte sich,
dass bestimmte Positivtypen systematisch übersehen wurden.

Diese schwierigen Positivbeispiele wurden anschließend:
- gespeichert
- über mehrere Seeds aggregiert
- und im WeightedRandomSampler stärker gewichtet.

Dadurch sollte das Modell gezielt lernen,
die bislang problematischen Muster besser zu erkennen.

Das Hard Positive Mining erwies sich unter Verwendundung des ResNet18 als nicht 
zielführend. Auffällig war auch die grosse Sicherheit mit der bestimmte Zebrastreifen-
Typen als `no` falsch klassifiziert wurden. Daraus wurde geschlossen, dass das Modell nicht  
leistungsfähig genug ist, die Unterschiede zu erkennen. 

---

# 20. Einsatz eines pretrained ResNet50-Modells

Nach den Experimenten mit eigenen CNN-Architekturen sowie dem pretrained ResNet18 wurde zusätzlich ein pretrained ResNet50 integriert. Ziel war es zu untersuchen, ob ein tieferes und leistungsfähigeres Netzwerk insbesondere die verbleibenden False Negatives besser erkennen kann.

### Motivation für ResNet50

Die Analyse der False Negatives zeigte, dass die verbleibenden Fehlerbilder häufig keine einfachen Fälle mehr waren. Viele dieser Bilder enthielten:

* starke Schatten
* komplexe Perspektiven
* teilweise verdeckte Zebrastreifen
* ungewöhnliche Straßenlayouts
* kleine oder fragmentierte Zebraflächen
* schwierige Beleuchtungssituationen

Das ResNet18 konnte bereits robuste allgemeine Merkmale lernen, wirkte jedoch teilweise noch zu konservativ gegenüber schwierigen Positivbeispielen.

Daher entstand die Hypothese, dass ein tieferes Modell:

* komplexere visuelle Strukturen besser erfassen kann,
* robustere hochlevelige Features lernt,
* und schwierige Positivmuster besser generalisiert.

### Architektur von ResNet50

Verwendet wurde:

```python
torchvision.models.resnet50
```

mit pretrained ImageNet-Gewichten.

Im Gegensatz zum ResNet18 verwendet ResNet50 sogenannte **Bottleneck Residual Blocks**.
Dadurch besitzt das Netzwerk:

* deutlich mehr Layer
* höhere Modellkapazität
* größere receptive fields
* komplexere Feature-Hierarchien

Die Residual-Verbindungen ermöglichen dabei weiterhin stabiles Training trotz großer Tiefe.

Die finale Klassifikationsschicht wurde analog zum ResNet18 ersetzt:

```python
self.backbone.fc = nn.Linear(in_features, num_classes)
```

### Fine-Tuning Strategie

Auch beim ResNet50 wurde ein zweistufiges Fine-Tuning verwendet:

#### Phase 1

* kompletter Backbone eingefroren
* nur der Classification Head trainiert

#### Phase 2

* letztes Residual-Block (`layer4`) freigegeben
* kleinere Learning Rate verwendet

Die kleinere Learning Rate erwies sich hier als besonders wichtig, da das grössere Modell empfindlicher auf aggressive Updates reagiert.

### Ergebnisse

Das ResNet50 führte zu einer weiteren Verbesserung der Ergebnisse.

Beispielsweise wurden Konfusionsmatrizen erreicht wie:
 
```text
[[6282   12]
 [  36  234]]
```

mit:

* Recall ≈ 0.84
* Precision ≈ 0.95–0.97
* F1 ≈ 0.89–0.90

Damit konnten die False Negatives gegenüber früheren Modellen nochmals reduziert werden.  

### Erkenntnisse aus der Fehleranalyse

Besonders interessant war die qualitative Veränderung der verbleibenden False Negatives.

Während frühe CNNs teilweise sehr einfache Zebrastreifen übersahen, bestanden die verbleibenden Fehlerbilder beim ResNet50 zunehmend aus tatsächlich schwierigen Fällen:

* starke Occlusions
* sehr kleine Zebraflächen
* extreme Perspektiven
* dominante Schattenbereiche
* ungewöhnliche urbane Strukturen
* Zebrastreifen anderer Farbe

Die False Negatives wirkten dadurch deutlich plausibler und weniger wie grundlegende Modellfehler.

Dies deutet darauf hin, dass das Modell:

* das allgemeine Konzept von Zebrastreifen inzwischen robust gelernt hat,
* die verbleibenden Fehler jedoch zunehmend durch echte visuelle Ambiguität entstehen.

### Auswirkungen auf Threshold-Tuning

Mit dem ResNet50 zeigte sich erneut eine sehr saubere Probability Separation.

Die Wahrscheinlichkeiten der Negativklasse lagen häufig extrem nahe bei 0, während Positivbeispiele hohe Konfidenzen erreichten. Dadurch wurde Threshold-Tuning zunehmend weniger kritisch.

Typisch waren stabile Ergebnisse über größere Threshold-Bereiche hinweg, was auf eine gute Probability Calibration hindeutet. 

### Einfluss auf die Seed-Stabilität

Ein weiterer wichtiger Befund war die deutlich reduzierte Seed-Abhängigkeit.

Im Vergleich zu den frühen CNN-Modellen schwankten:

* Recall,
* Precision,
* F1
* sowie die Konfusionsmatrizen

zwischen verschiedenen Seeds nur noch relativ gering.

Das pretrained ResNet50 erwies sich damit als:

* deutlich robuster,
* stabiler trainierbar,
* und weniger sensitiv gegenüber zufälliger Initialisierung oder Batch-Reihenfolge.

### Gesamtfazit zum ResNet50

Das ResNet50 stellte den bislang leistungsfähigsten Ansatz des Projekts dar.

Besonders verbessert wurden:

* Generalisierung,
* Stabilität,
* Probability Separation,
* und die Erkennung schwieriger Positivbeispiele.

Die verbleibenden Fehlerbilder wirken inzwischen deutlich plausibler und entsprechen eher echten Edge Cases als grundlegenden Modellschwächen.


# 21. Überfitting

Über weite Teile des Projekts bestand die Sorge vor starkem Overfitting.

Später zeigte sich jedoch:

```text
Das Hauptproblem war nicht klassisches Overfitting.
```

Sondern:

* Seed-Instabilität
* Datensatzabdeckung
* Modell ist zu konservativ

---

# 22. Zentrale Learnings des Projekts

### Seed-Abhängigkeit ist real

Einzelne gute Runs sind nicht zwangsläufig vertrauenswürdig.

### Learning Rate dominiert, vor allem unter Verwendung des SGD Optimizers

LR hatte grösseren Einfluss als:

* Scheduler
* Weight Decay
* kleinere Architekturänderungen

### Probability Calibration ist extrem wichtig

Gute Modelle:

* separieren Klassen klar
* machen Threshold weniger relevant
* stabilieren Modell

### Mehr Augmentation ist nicht automatisch besser

Zu aggressive Augmentation:

* zerstört semantische Muster
* verschlechtert Generalisierung

### Fehleranalyse ist entscheidend

Die Analyse der False Negatives war wahrscheinlich wertvoller als weiteres Hyperparameter-Tuning.
--> Verwendung von Hard Positives mehrerer Seeds für Training

### Pretrained Modelle helfen stark

ResNet50 lieferte:

* bessere Features
* robustere Entscheidungen
* sauberere Probability Separation

---

# 23. Aktuelle Einschätzung

Das Projekt hat sich von:

* instabilen kleinen CNNs
* starkem Klassenkollaps
* Recall-Problemen

hin entwickelt zu:

* robusteren pretrained Modellen
* guter Probability Separation
* sinnvoller Fehleranalyse
* systematischem Hyperparameter-Tuning

Der Fokus verschob sich weg von:

* Hyperparameter-Tuning

hin zu:

* gezielter Fehleranalyse
* Edge-Case-Abdeckung
* robuster Generalisierung
