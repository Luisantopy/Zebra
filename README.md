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


### Optimierungen: 
1. Weighted Random Sampling: *alpha* optimiert  
    *alpha* zu niedrig: hohe Accuracy aber Recall niedrig   ––> Sampler zu schwach
    *alpha* zu gross: hoher Recall aber Accuracy niedrig    ––> Sampler zu stark
    *alpha* = 0.90

2. Loss Function: *cross_entropy* und *binary_bce* verglichen
    *cross_entropy* leicht besser

3. Output Layer erweitern von Linear zu: 
    + Linear 128
    + ReLu
    + Linear 64

4. Threshold Tuning: 
    keine deutliche Verbesserung des F1 Scores, aber Recall fällt stark ab von 0.61 auf 0.38
    ––> kein sinnvoller Hebel, die Entscheidungsschwelle ist nicht das Kernproblem. 

### Bemerkungen: 
- wegen stark unbalancierter Klassen wurde WeightedRandomSampling verwendet; eine zusätzliche Gewichtung der Loss Funktion hat sich als kontraproduktiv erwiesen
- mit CrossEntropy Loss keine zusätzliche 'activation function' im Modell selber integriert