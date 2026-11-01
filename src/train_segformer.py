from pathlib import Path
import csv
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pretrained_segformer import SegFormer
from lidar_dataset import LidarDataset
from metrics import segmentation_metrics, calculate_metrics
SEED = 42
REPRESENTATION = 'combined'

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0
EXPERIMENT_NAME = 'Segformer_b0'
THRESHOLDS = [0.5, 0.5, 0.5]
CLASS_NAMES = ['Building', 'Platform', 'Aguada']

MANIFEST_PATH = PROJECT_ROOT / 'data/processed/manifest_split.csv'

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print('Device:', DEVICE)
train_dataset = LidarDataset(MANIFEST_PATH, split='train', representation=REPRESENTATION, augment=True)

val_dataset = LidarDataset(MANIFEST_PATH, split='val', representation=REPRESENTATION, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

print(f'Training tiles:', len(train_dataset))
print(f'Validation tiles:', len(val_dataset))

model = SegFormer(out_channels=3).to(DEVICE)

pos_weight = torch.tensor([1.0, 1.0, 10.0], device=DEVICE).view(3, 1, 1)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(f'Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS} Training', unit='batch', leave=False)

    for images, masks in progress:
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        n = images.size(0)

        running_loss += loss.item() * n
        total_samples += n

        progress.set_postfix(
            loss=f'{running_loss / total_samples:.4f}'
        )

    return running_loss / total_samples

@torch.no_grad()
def validate(model, loader, criterion, desc='Validation', thresholds=THRESHOLDS):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_tp = torch.zeros(3, device=DEVICE)
    total_fp = torch.zeros(3, device=DEVICE)
    total_fn = torch.zeros(3, device=DEVICE)

    for images, masks in tqdm(loader, desc=desc, unit='batch', leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        logits = model(images)
        loss = criterion(logits, masks)

        n = images.size(0)

        running_loss += loss.item() * n
        total_samples += n

        counts = segmentation_metrics(logits, masks, threshold=thresholds)

        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']

    results = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    results['loss'] = running_loss / total_samples

    return results

best_dice = -1.0
best_epoch = None
epochs_without_improvement = 0

history = []

checkpoint_path = (CHECKPOINT_DIR / f'{EXPERIMENT_NAME}.pt')

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)

    val_results = validate(model,val_loader,criterion)

    val_dice = val_results['mean_dice']
    val_iou = val_results['mean_iou']

    print(
        f'Epoch {epoch}/{EPOCHS} | '
        f'Train Loss: {train_loss:.4f} | '
        f'Val Loss: {val_results['loss']:.4f} | '
        f'Val Dice: {val_dice:.4f} | '
        f'Val IoU: {val_iou:.4f}'
    )

    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name}: 'f'Dice={val_results['dice'][i].item():.4f}, 'f'IoU={val_results['iou'][i].item():.4f}')

    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_results['loss'],
        'val_dice': val_dice,
        'val_iou': val_iou
    })

    if np.isfinite(val_dice) and val_dice > best_dice:

        best_dice = val_dice
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'representation': REPRESENTATION,
            'thresholds': THRESHOLDS,
            'experiment_name': EXPERIMENT_NAME,
            'best_val_dice': best_dice
        }, checkpoint_path)

        print('Saved new best SegFormer model.')

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print('Early stopping triggered.')
        break

history_path = (CHECKPOINT_DIR /f'{EXPERIMENT_NAME}_history.csv')

with open(history_path, 'w', newline='') as file:
    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )
    writer.writeheader()
    writer.writerows(history)

print('Training history saved:', history_path)

if best_epoch is None:
    raise RuntimeError(
        'No valid checkpoint was saved during training.'
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint['model_state_dict']
)

print(f'\nBEST CHECKPOINT: epoch {best_epoch}')

def print_results(title, results):
    print(f'\n{title}')
    print('----------------------')

    print(f'Loss: {results['loss']:.4f}')
    print(f'Mean Dice: {results['mean_dice']:.4f}')
    print(f'Mean IoU: {results['mean_iou']:.4f}')

    for i, name in enumerate(CLASS_NAMES):
        print(f'\n{name}')

        for metric in (
            'dice',
            'iou',
            'precision',
            'recall'
        ):
            print(f'{metric.capitalize()}: 'f'{results[metric][i].item():.4f}'
            )

best_val_results = validate(model, val_loader, criterion, desc='Best SegFormer Validation', thresholds=THRESHOLDS)

print_results('BEST SEGFORMER VALIDATION RESULTS',best_val_results)