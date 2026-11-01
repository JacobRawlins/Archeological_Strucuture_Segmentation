
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer import SegFormer
from lidar_dataset import LidarDataset
from metrics import segmentation_metrics, calculate_metrics

SEED = 42
REPRESENTATION = 'combined'

BATCH_SIZE = 2
NUM_WORKERS = 0

EXPERIMENT_NAME = 'Segformer_b0'

CLASS_NAMES = [
    'Building',
    'Platform',
    'Aguada'
]

MANIFEST_PATH = (PROJECT_ROOT / 'data/processed/manifest_split.csv')

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

CHECKPOINT_PATH = (CHECKPOINT_DIR /f'{EXPERIMENT_NAME}.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print('Device:', DEVICE)

val_dataset = LidarDataset(MANIFEST_PATH, split='val', representation=REPRESENTATION, augment=False)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

print('Validation tiles:', len(val_dataset))


model = SegFormer(out_channels=3).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

print('Loaded checkpoint from epoch:',checkpoint['epoch'])



THRESHOLD_VALUES = np.arange(
    0.05,
    0.951,
    0.025
)

best_thresholds = []
best_metrics = []


for class_idx, class_name in enumerate(CLASS_NAMES):

    print(f'\nTuning threshold for {class_name}')

    best_dice = -1.0
    best_threshold = 0.5
    best_result = None

    for threshold in tqdm(THRESHOLD_VALUES, desc=class_name):
        total_tp = torch.zeros(3, device=DEVICE)
        total_fp = torch.zeros(3, device=DEVICE)
        total_fn = torch.zeros(3, device=DEVICE)
        current_thresholds = [0.5, 0.5, 0.5]
        current_thresholds[class_idx] = float(threshold)
        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE, dtype=torch.float32)

                masks = masks.to(DEVICE,dtype=torch.float32)

                logits = model(images)

                counts = segmentation_metrics(logits, masks, threshold=current_thresholds)

                total_tp += counts['tp']
                total_fp += counts['fp']
                total_fn += counts['fn']

        results = calculate_metrics(total_tp, total_fp, total_fn)

        dice = results['dice'][class_idx].item()

        if np.isfinite(dice) and dice > best_dice:

            best_dice = dice
            best_threshold = float(threshold)
            best_result = {
                'dice': dice,
                'iou': results['iou'][class_idx].item(),
                'precision': results['precision'][class_idx].item(),
                'recall': results['recall'][class_idx].item()
            }

    best_thresholds.append(best_threshold)
    best_metrics.append(best_result)

    print(f'\nBest {class_name} Threshold: {best_threshold:.4f}')
    print(f'Dice: {best_result['dice']:.4f}')
    print(f'IoU: {best_result['iou']:.4f}')
    print(f'Precision: {best_result['precision']:.4f}')
    print(f'Recall: {best_result['recall']:.4f}')


print('\n===================================')
print('FINAL BEST THRESHOLDS')
print('===================================')

print(
    'BEST_THRESHOLDS =',
    [round(t, 4) for t in best_thresholds]
)

print('\nPer-Class Results:')

for class_name, result in zip(
    CLASS_NAMES,
    best_metrics
):
    print(
        f'{class_name}: '
        f'Dice={result['dice']:.4f}, '
        f'IoU={result['iou']:.4f}, '
        f'Precision={result['precision']:.4f}, '
        f'Recall={result['recall']:.4f}'
    )
