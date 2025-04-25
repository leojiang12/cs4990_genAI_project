import torch
import torch.nn as nn

# --- UNet generator skeleton ---
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        # Encoder: downsampling
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=False))
        self.down2 = nn.Sequential(nn.Conv2d(features, features*2, 4, 2, 1), nn.BatchNorm2d(features*2, track_running_stats=False), nn.LeakyReLU(0.2, inplace=False))
        # … add more downs as desired …
        # Decoder: upsampling
        self.up1   = nn.Sequential(nn.ConvTranspose2d(features*2, features, 4, 2, 1), nn.BatchNorm2d(features, track_running_stats=False), nn.ReLU(inplace=False))
        # Final output
        self.final = nn.Sequential(nn.ConvTranspose2d(features, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        return self.final(u1)

# --- PatchGAN Discriminator skeleton ---
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features, features*2, 4, 2, 1), nn.BatchNorm2d(features*2, track_running_stats=False), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(features*2, 1, 4, 1, 1)  # output 1-channel “real/fake” patch map
        )
    def forward(self, x, y):
        # concatenate input & target (or generated)
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)
