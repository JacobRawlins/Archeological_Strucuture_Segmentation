from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset 


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    if path.parts[0] == '..':
        path = Path(*path.parts[1:])

    return PROJECT_ROOT / path
def load_binary_mask(path):
        with rasterio.open(path ) as src:
            mask = src.read(1)
        mask = (mask == 0 )
        return mask

def load_tif(path):
        with rasterio.open(path) as src:
            array = src.read()
        return array

class LidarDataset(Dataset):
    def __init__(self, manifest_csv, split, representation = 'combined', augment = False):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest['split'] == split].reset_index(drop=True)
        self.representation = representation
        self.augment = augment

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image = load_tif(resolve_path(row['lidar_path']))
        building = load_binary_mask(resolve_path(row['building_mask']))
        platform = load_binary_mask(resolve_path(row['platform_mask']))
        aguada = load_binary_mask(resolve_path(row['aguada_mask']))
        if self.representation == 'svf':
            image = image[0:1]
        elif self.representation == 'openness':
            image = image[1:2]
        elif self.representation == 'slope':
            image = image[2:3]
        elif self.representation == 'combined':
            image = image
        else:
            raise ValueError(f'Unknown representation: {self.representation}')
        masks = np.stack([building, platform, aguada], axis=0).astype(np.float32)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        mask = masks.astype(np.float32)
        mask = torch.from_numpy(mask)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return image.contiguous(), mask.contiguous()
    



