
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.multi_attention_unet import MutliRepUnet
from lidar_dataset import LidarDataset

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

REPRESENTATION = 'combined'
BATCH_SIZE = 2
NUM_WORKERS = 0

REPRESENTATION_NAMES = ['SVF', 'Openness', 'Slope']

MANIFEST_PATH = (
    PROJECT_ROOT / 'data/processed/manifest_split.csv'
)

CHECKPOINT_PATH = (
    PROJECT_ROOT /
    'checkpoints/multi_rep_attention_unet_weighted_bce_best.pt'
)


# Validation dataset
val_dataset = LidarDataset(
    MANIFEST_PATH,
    split='val',
    representation=REPRESENTATION,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# Load trained model
model = MutliRepUnet(out_channels=3).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Loaded checkpoint epoch:', checkpoint['epoch'])


# Analyze attention
attention_sums = torch.zeros(4, 3, device=DEVICE)
attention_squares = torch.zeros(4, 3, device=DEVICE)
total_images = 0

with torch.no_grad():

    for images, masks in tqdm(
        val_loader,
        desc='Analyzing attention'
    ):
        images = images.to(DEVICE, dtype=torch.float32)

        logits, attention_weights = model(
            images,
            return_attention=True
        )

        batch_size = images.size(0)
        total_images += batch_size

        for level, weights in enumerate(attention_weights):
            # [B, 3, 1, 1] -> [B, 3]
            weights = weights.squeeze(-1).squeeze(-1)

            attention_sums[level] += weights.sum(dim=0)
            attention_squares[level] += (
                weights ** 2
            ).sum(dim=0)


attention_means = attention_sums / total_images

attention_stds = torch.sqrt(
    (
        attention_squares / total_images
        - attention_means ** 2
    ).clamp(min=0)
)

print('\nAVERAGE ATTENTION BY ENCODER LEVEL')
print('----------------------------------')

for level in range(4):
    print(f'\nEncoder Level {level + 1}')

    for c, name in enumerate(REPRESENTATION_NAMES):
        print(
            f'  {name}: '
            f'Mean={attention_means[level, c].item():.4f}, '
            f'Std={attention_stds[level, c].item():.4f}'
        )

print('\nOVERALL AVERAGE ATTENTION')
print('----------------------------------')

overall_attention = attention_means.mean(dim=0)

for c, name in enumerate(REPRESENTATION_NAMES):
    print(
        f'{name}: {overall_attention[c].item():.4f}'
    )
