import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

TEST_RUN = False


class SegFormer(nn.Module):
    def __init__(self, out_channels = 3):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512', num_labels = out_channels, ignore_mismatched_sizes=True)

    def forward(self, x):
        outputs = self.model(pixel_values = x)
        logits = outputs.logits

        logits = F.interpolate(logits, size = x.shape[2:], mode = 'bilinear', align_corners =False)
        return logits





if TEST_RUN:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegFormer(out_channels = 3).to(device)
    x = torch.randn(2,3,480,480).to(device)

    with torch.no_grad():
        logits = model(x)

    print(f'Input Shape: {x.shape}')
    print(f'Output Shape: {logits.shape}')
    print(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    