import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.pre = self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1), nn.PReLU(16)
        )
        self.conv00 = BasicBlock(16, 32)
        self.down0 = nn.MaxPool2d((2, 2))
        self.conv10 = BasicBlock(32, 64)
        self.down1 = nn.MaxPool2d((2, 2))
        self.conv20 = BasicBlock(64, 128)
        self.down2 = nn.MaxPool2d((2, 2))
        self.conv30 = BasicBlock(128, 256)
        self.conv31 = BasicBlock(256, 512)
        self.up2 = nn.PixelShuffle(2)
        self.conv21 = BasicBlock(128, 256)
        self.up1 = nn.PixelShuffle(2)
        self.conv11 = BasicBlock(64, 128)
        self.up0 = nn.PixelShuffle(2)
        self.conv01 = BasicBlock(32, 64)

        self.post = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x0 = self.pre(x)
        x0 = self.conv00(x0)
        x1 = self.down0(x0)
        x1 = self.conv10(x1)
        x2 = self.down1(x1)
        x2 = self.conv20(x2)
        x3 = self.down2(x2)
        x3 = self.conv30(x3)
        x3 = self.conv31(x3)
        x2 = x2 + self.up2(x3)
        x2 = self.conv21(x2)
        x1 = x1 + self.up1(x2)
        x1 = self.conv11(x1)
        x0 = x0 + self.up0(x1)
        x0 = self.conv01(x0)
        x = self.post(x0)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels, kernel_size=3, padding=1, stride=1
                    ),
                    nn.PReLU(in_channels),
                )
            )
        self.main = nn.Sequential(*layers)
        self.post = nn.Sequential(
            nn.Conv2d(
                in_channels * 4, out_channels, kernel_size=3, padding=1, stride=1
            ),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        out = []
        out.append(x)
        for layers in self.main:
            x = layers(x)
            out.append(x)
        x = torch.concat(out, axis=1)
        x = self.post(x)
        return x


if __name__ == "__main__":
    model = UNet()
    input = torch.rand(size=(16, 3, 384, 512))
    output = model(input)
    print(output.shape)
