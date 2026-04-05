import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from helpers import seed_worker


class ConditionalImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform_n=None, transform_y=None):
        super().__init__(root=root)
        self.transform_n = transform_n
        self.transform_y = transform_y

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        if label == 1:  # y
            if self.transform_y is not None:
                image = self.transform_y(image)
        else:  # n
            if self.transform_n is not None:
                image = self.transform_n(image)

        return image, label


def get_train_dataset(root="data/train"):
    """Trainingsdaten laden, y stärker augmentieren als n"""
    transform_n = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.1),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_y = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
        v2.RandomPerspective(p=0.2),
        v2.RandomRotation(20),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ConditionalImageFolder(
        root=root,
        transform_n=transform_n,
        transform_y=transform_y
    )


def get_eval_dataset(root="data/val"):
    """Validierungsdaten nur normalisieren, nicht augmentieren"""
    eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return datasets.ImageFolder(
        root=root,
        transform=eval_transforms,
    )


def get_loader(dataset, batch_size=32, shuffle=False, sampler=None, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g
    )