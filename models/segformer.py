
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerConfig, SegformerForSemanticSegmentation

TEST_RUN = False


class SegFormer(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()


        config = SegformerConfig.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')

        config.num_labels = out_channels

        config.id2label = {
            0: 'Building',
            1: 'Platform',
            2: 'Aguada'
        }

        config.label2id = {
            'Building': 0,
            'Platform': 1,
            'Aguada': 2
        }


        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        logits = F.interpolate(logits,size=x.shape[2:],mode='bilinear',align_corners=False)
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

    