
from pathlib import Path
import csv
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.multi_attention_unet import MutliRepUnet
from src.lidar_dataset import LidarDataset
from src.metrics import segmentation_metrics, calculate_metrics


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPRESENTATION = 'combined'
BATCH_SIZE = 2
NUM_WORKERS = 0

CLASS_NAMES = ['Building', 'Platform', 'Aguada']

MANIFEST_PATH = PROJECT_ROOT / 'data/processed/manifest_split.csv'

CHECKPOINT_PATH = (PROJECT_ROOT /'checkpoints/multi_rep_unet.pt')

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def tune_thresholds(model, loader):
    model.eval()

    thresholds = [round(float(t), 4) for t in torch.arange(0.05, 0.951, 0.025)]

    tp = torch.zeros(len(thresholds), 3, device=DEVICE)
    fp = torch.zeros(len(thresholds), 3, device=DEVICE)
    fn = torch.zeros(len(thresholds), 3, device=DEVICE)

    for images, masks in tqdm(loader, desc='Threshold tuning', unit='batch'
    ):
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)
        logits = model(images)
        for i, threshold in enumerate(thresholds):
            counts = segmentation_metrics(logits, masks, threshold=[threshold] * 3)
            tp[i] += counts['tp']
            fp[i] += counts['fp']
            fn[i] += counts['fn']

    best_thresholds = []
    best_dice = [-1.0] * 3
    rows = []

    for i, threshold in enumerate(thresholds):
        results = calculate_metrics(tp[i], fp[i], fn[i])
        for c, name in enumerate(CLASS_NAMES):
            dice = results['dice'][c].item()
            iou = results['iou'][c].item()
            precision = results['precision'][c].item()
            recall = results['recall'][c].item()

            rows.append({
                'class': name,
                'threshold': threshold,
                'dice': dice,
                'iou': iou,
                'precision': precision,
                'recall': recall
            })

            if dice > best_dice[c]:
                best_dice[c] = dice
                if len(best_thresholds) <= c:
                    best_thresholds.extend(
                        [0.5] * (c + 1 - len(best_thresholds))
                    )
                best_thresholds[c] = threshold

    for c, name in enumerate(CLASS_NAMES):
        selected = next(
            row for row in rows
            if row['class'] == name
            and row['threshold'] == best_thresholds[c]
        )

        print(
            f'{name}: '
            f"Threshold={selected['threshold']:.4f}, "
            f"Dice={selected['dice']:.4f}, "
            f"IoU={selected['iou']:.4f}, "
            f"Precision={selected['precision']:.4f}, "
            f"Recall={selected['recall']:.4f}"
        )

    output_path = (
        RESULTS_DIR /
        f'threshold_search_multi_rep_attention_{REPRESENTATION}.csv'
    )

    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'class',
                'threshold',
                'dice',
                'iou',
                'precision',
                'recall'
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print('\nThreshold results saved:', output_path)

    return best_thresholds
val_dataset = LidarDataset(MANIFEST_PATH, split='val', representation=REPRESENTATION, augment=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = MutliRepUnet(out_channels=3).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Loaded checkpoint:', CHECKPOINT_PATH)
print('Checkpoint epoch:', checkpoint['epoch'])
best_thresholds = tune_thresholds(model, val_loader)

print('\nBest thresholds:', best_thresholds)
