from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pretrained_segformer import SegFormer
from src.lidar_dataset import LidarDataset
from src.metrics import segmentation_metrics, calculate_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPRESENTATION = 'combined'
BATCH_SIZE = 2
NUM_WORKERS = 0

EXPERIMENT_NAME = 'Segformer_b0'

CLASS_NAMES = ['Building', 'Platform', 'Aguada']

BEST_THRESHOLDS = [0.425, 0.4, 0.5]

MANIFEST_PATH = (PROJECT_ROOT / 'data/processed/manifest_split.csv')

CHECKPOINT_PATH = (PROJECT_ROOT/ 'checkpoints'/ f'{EXPERIMENT_NAME}.pt')



test_dataset = LidarDataset(MANIFEST_PATH, split='test', representation=REPRESENTATION, augment=False)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print('Device:', DEVICE)
print('Test tiles:', len(test_dataset))


model = SegFormer(out_channels=3).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Loaded checkpoint:', CHECKPOINT_PATH)
print('Checkpoint epoch:', checkpoint['epoch'])
print('Thresholds:', BEST_THRESHOLDS)


pos_weight = torch.tensor([1.0, 1.0, 10.0], device=DEVICE).view(3, 1, 1)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

@torch.no_grad()
def evaluate_test(model, loader, criterion, thresholds):

    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_tp = torch.zeros(3, device=DEVICE)
    total_fp = torch.zeros(3, device=DEVICE)
    total_fn = torch.zeros(3, device=DEVICE)

    for images, masks in tqdm(
        loader,
        desc='Testing SegFormer',
        unit='batch'
    ):
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        logits = model(images)

        loss = criterion(logits, masks)
        n = images.size(0)
        running_loss += loss.item() * n
        total_samples += n
        counts = segmentation_metrics(logits, masks, threshold=thresholds)

        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']

    results = calculate_metrics(total_tp, total_fp, total_fn)

    results['loss'] = running_loss / total_samples

    return results

results = evaluate_test(
    model,
    test_loader,
    criterion,
    BEST_THRESHOLDS
)

print('\nBEST SEGFORMER TEST RESULTS')
print('-------------------------------------------')
print(f'Loss: {results['loss']:.4f}')
print(f'Mean Dice: {results['mean_dice']:.4f}')
print(f'Mean IoU: {results['mean_iou']:.4f}')

for i, name in enumerate(CLASS_NAMES):
    print(f'\n{name}')
    print(f'Dice: {results['dice'][i].item():.4f}')
    print(f'IoU: {results['iou'][i].item():.4f}')
    print(f'Precision: {results['precision'][i].item():.4f}')
    print(f'Recall: {results['recall'][i].item():.4f}')