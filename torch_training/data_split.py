from torchvision import datasets
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


dataset = datasets.ImageFolder("data/raw_data")
targets = [label for _, label in dataset.samples]

indices = list(range(len(dataset)))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.3,
    stratify=targets,
    random_state=42
)

temp_targets = [targets[i] for i in temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_targets,
    random_state=42
)

base_out = Path("data")

for split_name, split_indices in {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}.items():
    for idx in split_indices:
        path, label = dataset.samples[idx]
        class_name = dataset.classes[label]

        out_dir = base_out / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(path, out_dir / Path(path).name)