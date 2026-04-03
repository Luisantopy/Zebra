# Zebra
Binary image classification of crosswalks


Folgende Ordnerstruktur wird vom Modell erwartet: 
    data/
        aug/
        raw_data/
            y/
            n/
        test/
        train/
        val/
    trained_models/

Daten vorbereiten: 
- Rohdaten müssen im Ordner raw_data liegen
- 1x data_split.py laufen lassen, um die Bilder aus raw_data in train, val und test aufzuteilen

Modell trainieren: 
uv run python -m training_pipeline
--> Trainierte Modelle werden inkl. config Dateien automatisch in einen eigenen Unterordner in trained_models abgelegt. 


Einzelne Vorhersage machen: 
uv run python -m predict_single "path/zum/bild.png" --weights trained_models/"experiment_folder"/best_model.pth
uv run python predict_single.py data/test/y/2758700_1191950.png --weights trained_models/exp_20260403_174647/best_model.pth --model binary_bce --classes n y
uv run python predict_single.py data/test/y/2758700_1191950.png --weights trained_models/exp_20260403_143400/best_model.pth --model cross_entropy --classes n y


Bemerkungen: 
- wegen stark unbalancierter Klassen wurde WeightedRandomSampling verwendet; eine zusätzliche Gewichtung der Loss Funktion hat sich als kontraproduktiv erwiesen
- mit CrossEntropy Loss keine zusätzliche 'activation function' im Modell selber integriert