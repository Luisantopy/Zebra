import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_keypoints, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


# --- image plot helper from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            points = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target

                    # Conversion necessary because draw_bounding_boxes() only
                    # work with this specific format.
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat("xyxyxyxy")(boxes)
                elif isinstance(target, tv_tensors.KeyPoints):
                    points = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)
            if points is not None:
                img = draw_keypoints(img, points, colors="red", radius=10)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# --- Threshold helpers ---
@torch.no_grad()
def evaluate_thresholds(model, loader, device, thresholds=None, positive_label=1):
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    model.eval()
    results = []

    for threshold in thresholds:
        correct = 0
        total = 0

        true_positive = 0
        false_negative = 0
        false_positive = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Wahrscheinlichkeit für Klasse 1 = "y"
            probs = torch.softmax(outputs, dim=1)[:, positive_label]

            preds = (probs >= threshold).long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            true_positive += ((preds == positive_label) & (labels == positive_label)).sum().item()
            false_negative += ((preds != positive_label) & (labels == positive_label)).sum().item()
            false_positive += ((preds == positive_label) & (labels != positive_label)).sum().item()

        acc = correct / total if total > 0 else 0.0

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append({
            "threshold": threshold,
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

    return results


@torch.no_grad()
def evaluate_with_threshold(model, loader, device, threshold=0.5, positive_label=1):
    model.eval()

    correct = 0
    total = 0

    true_positive = 0
    false_negative = 0
    false_positive = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, positive_label]
        preds = (probs >= threshold).long()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        true_positive += ((preds == positive_label) & (labels == positive_label)).sum().item()
        false_negative += ((preds != positive_label) & (labels == positive_label)).sum().item()
        false_positive += ((preds == positive_label) & (labels != positive_label)).sum().item()

    acc = correct / total if total > 0 else 0.0

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return acc, recall, precision, f1