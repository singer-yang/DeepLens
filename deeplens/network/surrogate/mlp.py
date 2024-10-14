import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """All-linear layer. This network suits for low-k intensity/amplitude PSF function prediction."""

    def __init__(self, in_features, out_features, hidden_features=64, hidden_layers=3):
        super(MLP, self).__init__()

        layers = [
            nn.Linear(in_features, hidden_features // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features // 4, hidden_features, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_features, hidden_features, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.extend(
            [nn.Linear(hidden_features, out_features, bias=True), nn.Sigmoid()]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, p=1, dim=-1)
        return x


if __name__ == "__main__":
    # Test the network
    mlp = MLP(4, 64, hidden_features=64, hidden_layers=3)
    print(mlp)
    x = torch.rand(100, 4)
    y = mlp(x)
    print(y.size())
