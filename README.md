# Zebra
Binary image classification of crosswalks


### Folgende Ordnerstruktur wird vom Modell erwartet: 
    data/
        aug/
        raw_data/
            y/
            n/
        test/
        train/
        val/
    trained_models/

### Daten vorbereiten: 
- Rohdaten müssen im Ordner raw_data liegen
- 1x data_split.py laufen lassen, um die Bilder aus raw_data in train, val und test aufzuteilen

### Modell trainieren: 
uv run python -m training_pipeline
--> Trainierte Modelle werden inkl. config + metrics Dateien automatisch in einen eigenen Unterordner in trained_models abgelegt


### Einzelne Vorhersage machen: 
uv run python -m predict_single "path/zum/bild.png" --weights trained_models/"experiment_folder"/best_model.pth
uv run python predict_single.py data/test/y/2758700_1191950.png --weights trained_models/exp_20260403_174647/best_model.pth --model binary_bce --classes n y
uv run python predict_single.py data/test/y/2758700_1191950.png --weights trained_models/exp_20260403_143400/best_model.pth --model cross_entropy --classes n y


### Seed setzen
    Ziel: Training stabilisieren


### Optimierungen mit Optuna: 
1. Weighted Random Sampling: 
    Ziel: Klassenungleichgweicht entgegensteuern
    ––> Sampler zu schwach dh. *alpha* zu niedrig: hohe Accuracy aber Recall niedrig   
    ––> Sampler zu stark dh. *alpha* zu gross: hoher Recall aber Accuracy niedrig    
    best *alpha* = 0.90
    Test Loss: 0.3201 | Test Acc: 0.8286 | Test Recall: 0.5614 | Test Precision: 0.0549 | Test F1: 0.1000

2. Optimizer + Learning Rate
    Ziel: Optimizer Funktion für bessere Balancierung der Metriken finden
    Optimizer Vergleich: *Adam* und *SGD* (jeweils mit Learning Rate Optimierungen)
    *Adam*: keine Verbesserung durch Anpassen der Lernrate
    *SGD*: deutliche Verbesserung ggüber Adam
    Test with tuned threshold=0.45 | Acc: 0.9842 | Recall: 0.3158 | Precision: 0.5625 | F1: 0.4045

### Manuelle Optimierungen: 
5. Data Augmentation:
    Ziel: Klassenungleichgewicht entgegensteuern
    asymmetrische Augmentationen für y/ und n/ Klassen mit dem Ziel Ungleichgewicht entgegenzusteuern und Precision zu erhöhen
    ––> keine zu schwache Augmentation
    ––> asymmetrische Augmentation hilft, darf aber nicht zu stark sein
    ––> gezieltes Feintuning der Data Augmentation verbessert das Resultat deutlich
    ––> ab hier ist Threshold Tuning sinnvoll
    Test with tuned threshold=0.50 | Acc: 0.9774 | Recall: 0.2281 | Precision: 0.2889 | F1: 0.2549

3. Architektur:
    ––> Output Layer:
        Ziel: mehr Kapazität im Head
        erweitern von Linear zu: 
        + Linear 128
        + ReLu
        + Linear 64
        Test Loss: 0.3472 | Test Acc: 0.7688 | Test Recall: 0.8246 | Test Precision: 0.0577 | Test F1: 0.1079
    ––> Hidden Layers: 
        Ziel: Verbesserung der Generalisierung, 

3. Loss Function: 
    Ziel: Verlustfunktion an Problemstellung anpassen, Balance aus Precision und Accuracy verbessern
    *cross_entropy*:    Test Loss: 0.3201 | Test Acc: 0.8286 | Test Recall: 0.5614 | Test Precision: 0.0549 | Test F1: 0.1000
    *binary_bce*:       Test Loss: 1.1073 | Test Acc: 0.6223 | Test Recall: 0.9649 | Test Precision: 0.0416 | Test F1: 0.0798
    ––> *cross_entropy* leicht besser, *binary_bce* wird erstmal nicht weiterverfolgt

4. Threshold Tuning auf Validation Data (min_recall): 
    Ziel: Optimieren der Entscheidungsgrenze
    ––> sinnvoller Hebel, wenn Modell gut genug 
    
### Bemerkungen: 
- wegen stark unbalancierter Klassen wurde WeightedRandomSampling verwendet; eine zusätzliche Gewichtung der Loss Funktion hat sich als kontraproduktiv erwiesen
- mit CrossEntropy Loss keine zusätzliche 'activation function' im Modell selber integriert