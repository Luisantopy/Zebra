import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_train_dataset(root="data/train"):
    """Trainingsdaten beim Laden augmentieren"""
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.2),
        v2.GaussianBlur(kernel_size=(3, 7)),
        v2.ColorJitter(brightness=0.5, contrast=0.7, saturation=0.8),
        v2.RandomPerspective(p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return datasets.ImageFolder(
        root=root,
        transform=train_transforms,
    )


def get_eval_dataset(root="data/val"):
    """Validierungsdaten beim Laden nur ToTensor/Normalize"""
    eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return datasets.ImageFolder(
        root=root,
        transform=eval_transforms,
    )


def get_loader(dataset, batch_size=32, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )