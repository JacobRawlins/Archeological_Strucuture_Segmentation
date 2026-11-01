from pathlib import Path
import csv
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet
from src.lidar_dataset import LidarDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPRESENTATION = 'combined'
BATCH_SIZE = 2
NUM_WORKERS = 0

EXPERIMENT_NAME = 'unet_combined'

CLASS_NAMES = ['Building', 'Platform', 'Aguada']

MANIFEST_PATH = PROJECT_ROOT / 'data/processed/manifest_split.csv'

CHECKPOINT_PATH = (PROJECT_ROOT / 'checkpoints' / f'{EXPERIMENT_NAME}.pt')

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def tune_thresholds(model, loader):
    model.eval()

    thresholds = torch.arange(0.05, 0.951, 0.025).tolist()

    tp = torch.zeros(len(thresholds), 3, device=DEVICE)
    fp = torch.zeros(len(thresholds), 3, device=DEVICE)
    fn = torch.zeros(len(thresholds), 3, device=DEVICE)

    for images, masks in tqdm(
        loader,
        desc='Threshold tuning',
        unit='batch'
    ):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).bool()

        probs = torch.sigmoid(model(images))

        for i, threshold in enumerate(thresholds):
            preds = probs >= threshold

            tp[i] += (preds & masks).sum(dim=(0, 2, 3))
            fp[i] += (preds & ~masks).sum(dim=(0, 2, 3))
            fn[i] += (~preds & masks).sum(dim=(0, 2, 3))

    dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-7)
    iou = tp / (tp + fp + fn).clamp(min=1e-7)
    precision = tp / (tp + fp).clamp(min=1e-7)
    recall = tp / (tp + fn).clamp(min=1e-7)

    best_indices = dice.argmax(dim=0)
    best_thresholds = []
    rows = []

    for i, threshold in enumerate(thresholds):
        for c, name in enumerate(CLASS_NAMES):
            rows.append({
                'class': name,
                'threshold': round(threshold, 4),
                'dice': dice[i, c].item(),
                'iou': iou[i, c].item(),
                'precision': precision[i, c].item(),
                'recall': recall[i, c].item()
            })

    for c, name in enumerate(CLASS_NAMES):
        idx = best_indices[c].item()
        threshold = thresholds[idx]
        best_thresholds.append(threshold)

        print(
            f'{name}: '
            f'Threshold={threshold:.3f}, '
            f'Dice={dice[idx, c].item():.4f}, '
            f'IoU={iou[idx, c].item():.4f}, '
            f'Precision={precision[idx, c].item():.4f}, '
            f'Recall={recall[idx, c].item():.4f}'
        )

    output_path = (RESULTS_DIR/ f'{EXPERIMENT_NAME}_threshold_search.csv')

    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print('\nThreshold search saved to:', output_path)

    return best_thresholds


val_dataset = LidarDataset(MANIFEST_PATH, split='val', representation=REPRESENTATION, augment=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


model = UNet(in_channels=3, out_channels=3).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

print('Loaded checkpoint:', CHECKPOINT_PATH)
print('Checkpoint epoch:', checkpoint['epoch'])

best_thresholds = tune_thresholds(model, val_loader)

print('\nBEST U-NET THRESHOLDS:')
print([round(t, 3) for t in best_thresholds])