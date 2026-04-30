
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlusDecoder(nn.Module):
    """
    Proper UNet++ decoder for TransUNet++.

    Receives:
        hidden_states : (B, n_patches, hidden_dim)  — ViT encoder output
        features      : list of 3 skip tensors from ResNetV2, coarse-first
                        [block3_out, block2_out, block1_out]
                        channels (width_factor=1): [1024, 512, 256]

    Architecture (for 224px input, width_factor=1):
        ViT tokens → reshape → proj → x00  ( 512 ch,  14x14)
        x10 = up(x00) cat skip[0](1024ch) → ConvBlock → ( 512 ch,  28x28)
        x20 = up(x10) cat skip[1]( 512ch) → ConvBlock → ( 256 ch,  56x56)
        x30 = up(x20) cat skip[2]( 256ch) → ConvBlock → ( 128 ch, 112x112)
        x40 = up(x30)                      → ConvBlock → (  64 ch, 224x224)

    Returns:
        x40  : (B,  64, H,   W)   ← fed to primary SegmentationHead
        [x30, x20, x10]           ← fed to aux deep-supervision heads
        channels: [128, 256, 512]
    """

    # Internal channel widths at each decoder level
    CH = [512, 512, 256, 128, 64]

    def __init__(self, config):
        super().__init__()

        hidden = config.hidden_size   # 768 for ViT-B

        # Skip channels from ResNetV2 coarse→fine: block3, block2, block1
        # For width_factor=1: [1024, 512, 256]
        # For width_factor=2: [2048, 1024, 512]
        wf = getattr(getattr(config, "resnet", None), "width_factor", 1)
        base = int(64 * wf)
        self.skip_ch = [base * 16, base * 8, base * 4]   # [1024, 512, 256]

        # Project ViT tokens → first decoder width (512)
        self.proj = nn.Sequential(
            nn.Conv2d(hidden, self.CH[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.CH[0]),
            nn.ReLU(inplace=True),
        )

        # block1: (CH[0] + skip_ch[0]) → CH[1]
        self.dec1 = ConvBlock(self.CH[0] + self.skip_ch[0], self.CH[1])
        # block2: (CH[1] + skip_ch[1]) → CH[2]
        self.dec2 = ConvBlock(self.CH[1] + self.skip_ch[1], self.CH[2])
        # block3: (CH[2] + skip_ch[2]) → CH[3]
        self.dec3 = ConvBlock(self.CH[2] + self.skip_ch[2], self.CH[3])
        # block4: CH[3] → CH[4]  (no skip at this resolution)
        self.dec4 = ConvBlock(self.CH[3], self.CH[4])

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _align(self, feat, target):
        """Resize feat spatially to match target if needed."""
        if feat.shape[2:] != target.shape[2:]:
            feat = F.interpolate(feat, size=target.shape[2:],
                                 mode="bilinear", align_corners=False)
        return feat

    def forward(self, hidden_states, features=None):
        B, n, c = hidden_states.shape
        h = w = int(n ** 0.5)

        # Reshape ViT tokens to spatial map
        x = hidden_states.permute(0, 2, 1).view(B, c, h, w)
        x00 = self.proj(x)                          # (B, 512, 14, 14)

        # Unpack skip connections  (coarse-first)
        if features is not None and len(features) >= 3:
            sk0, sk1, sk2 = features[0], features[1], features[2]
        else:
            # No hybrid backbone — create zero skips of the correct shape
            sk0 = torch.zeros(B, self.skip_ch[0], h,    w,    device=x.device)
            sk1 = torch.zeros(B, self.skip_ch[1], h*2,  w*2,  device=x.device)
            sk2 = torch.zeros(B, self.skip_ch[2], h*4,  w*4,  device=x.device)

        # Level 1:  14 → 28
        x10_up = self.up(x00)                       # (B, 512, 28, 28)
        sk0    = self._align(sk0, x10_up)
        x10    = self.dec1(torch.cat([x10_up, sk0], dim=1))   # (B, 512, 28, 28)

        # Level 2:  28 → 56
        x20_up = self.up(x10)                       # (B, 512, 56, 56)
        sk1    = self._align(sk1, x20_up)
        x20    = self.dec2(torch.cat([x20_up, sk1], dim=1))   # (B, 256, 56, 56)

        # Level 3:  56 → 112
        x30_up = self.up(x20)                       # (B, 256, 112, 112)
        sk2    = self._align(sk2, x30_up)
        x30    = self.dec3(torch.cat([x30_up, sk2], dim=1))   # (B, 128, 112, 112)

        # Level 4:  112 → 224  (no skip)
        x40    = self.dec4(self.up(x30))             # (B,  64, 224, 224)

        # aux outputs for deep supervision, coarse→fine
        # channels: x30=128, x20=256, x10=512
        return x40, [x30, x20, x10]
