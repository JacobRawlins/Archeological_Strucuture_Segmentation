import torch
import torch.nn as nn
import torch.nn.functional as F


TEST_RUN = False


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, 3, kernel_size = 1)
        )
    def forward(self, svf, openness, slope):
        combined = torch.cat([svf, openness, slope], dim = 1)
        weights = self.attention(combined)
        weights = torch.softmax(weights, dim = 1)
        fused = (weights[:, 0:1] * svf + weights[:, 1:2] * openness + weights[:, 2:3] * slope)
        return fused, weights

class UpSample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size = skip.shape[2:], mode = 'bilinear', align_corners = False)
        x = torch.cat([x, skip], dim = 1)
        return self.conv(x)


class MutliRepUnet(nn.Module):
    def __init__(self, out_channels = 3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.svf_enc1 = DoubleConv(1,32)
        self.svf_enc2 = DoubleConv(32, 64)
        self.svf_enc3 = DoubleConv(64, 128)
        self.svf_enc4 = DoubleConv(128, 256)
        self.open_enc1 = DoubleConv(1,32)
        self.open_enc2 = DoubleConv(32,64)
        self.open_enc3 = DoubleConv(64, 128)
        self.open_enc4 = DoubleConv(128,256)
        self.slope_enc1 = DoubleConv(1,32)
        self.slope_enc2 = DoubleConv(32,64)
        self.slope_enc3 = DoubleConv(64,128)
        self.slope_enc4 = DoubleConv(128,256)

        self.fusion1 = AttentionFusion(32)
        self.fusion2 = AttentionFusion(64)
        self.fusion3 = AttentionFusion(128)
        self.fusion4 = AttentionFusion(256)
        self.bottleneck = DoubleConv(256,512)

        self.up4 = UpSample(512,256,256)
        self.up3 = UpSample(256,128,128)
        self.up2 = UpSample(128,64,64)
        self.up1 = UpSample(64,32,32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size = 1)

    def forward(self, x, return_attention = False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('this should be a 4 value tensor with [Batch, 3, H, W]')
        svf = x[:,0:1, :, :]
        openness = x[:, 1:2, :, :]
        slope = x[:, 2:3, :, :]
        s1 = self.svf_enc1(svf)
        o1 = self.open_enc1(openness)
        p1 = self.slope_enc1(slope)
        f1, a1 = self.fusion1(s1, o1, p1)
        s2 = self.svf_enc2(self.pool(s1))
        o2 = self.open_enc2(self.pool(o1))
        p2 = self.slope_enc2(self.pool(p1))

        f2, a2 = self.fusion2(s2, o2, p2)

        s3 = self.svf_enc3(self.pool(s2))
        o3 = self.open_enc3(self.pool(o2))
        p3 = self.slope_enc3(self.pool(p2))

        f3, a3 = self.fusion3(s3, o3, p3)
        s4 = self.svf_enc4(self.pool(s3))
        o4 = self.open_enc4(self.pool(o3))
        p4 = self.slope_enc4(self.pool(p3))
        f4, a4 = self.fusion4(s4, o4, p4)
        b = self.bottleneck(self.pool(f4))
        d4 = self.up4(b, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        logits = self.final_conv(d1)
        if return_attention:
            attention_weights = [a1, a2, a3, a4]
            return logits, attention_weights
        return logits



if TEST_RUN:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MutliRepUnet(out_channels = 3).to(device)
    x = torch.randn(1, 3, 480, 480).to(device)
    with torch.no_grad():
        output = model(x)
    print(f'Input Shape: {x.shape}')
    print(f'Output Shape: {output.shape}')
