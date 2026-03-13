import torch.nn as nn
import torch
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(out_channels)
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv_res(x)
        x = self.net(x)
        return self.relu(x + res)

class TransposedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, 2) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y = torch.meshgrid([x, x], indexing='ij')
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid

class ResNetConditionalParameterRegressor(nn.Module):
    def __init__(self, num_parts):
        super(ResNetConditionalParameterRegressor, self).__init__()
        """
        convolutional encoder + linear layer at the end
        Args:
            num_features: list of ints containing number of features per layer
            num_parts: number of body parts for which we regress affine parameters
        Returns:
            torch.tensor (batch, num_parts, 2, 3), (2, 3) affine matrix for each body part
        """
        self.num_parts = num_parts
        self.output_size = 32
        self.conv = nn.Sequential(
            ResBlock(3, 64),  # 64
            ResBlock(64, 128),  # 32
            ResBlock(128, 256),  # 16
            ResBlock(256, 512),  # 8
            TransposedBlock(512, 256),  # 16
            TransposedBlock(256, 128),  # 32
        )
        self.pred_prob = nn.Conv2d(128, self.num_parts, kernel_size=3, padding=1)
        grid = gen_grid2d(self.output_size).reshape(1, self.output_size ** 2, 2)
        self.coord = nn.Parameter(grid, requires_grad=False)
        self.emb = nn.Linear(2, 128)
        self.predict = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 6)
        )

    def forward(self, input, template_points):
        B = input.shape[0]
        prob_feat = self.conv(input)  # (B, 128, h, w)
        prob_map = self.pred_prob(prob_feat)  # (B, n, h, w)
        prob_map = F.softmax(prob_map.flatten(2, 3), dim=2)
        key_feat = torch.matmul(prob_map, torch.cat([prob_feat.flatten(2, 3).transpose(1, 2), self.emb(self.coord.repeat(B, 1, 1))], dim=-1))
        return self.predict(torch.cat([key_feat, self.emb(template_points)], dim=-1).flatten(0, 1)).view(B, self.num_parts, 2, 3)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class ResNetReconstructor(nn.Module):
    def __init__(self):
        super(ResNetReconstructor, self).__init__()
        """
        convolutional encoder, decoder
        """
        self.down0 = nn.Sequential(
            nn.Conv2d(3 + 1, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down1 = DownBlock(64, 128)  # 64
        self.down2 = DownBlock(128, 256)  # 32
        self.down3 = DownBlock(256, 512)  # 16
        self.down4 = DownBlock(512, 512)  # 8

        self.up1 = UpBlock(512, 512)  # 16
        self.up2 = UpBlock(512 + 512, 256)  # 32
        self.up3 = UpBlock(256 + 256, 128)  # 64
        self.up4 = UpBlock(128 + 128, 64)  # 64

        self.conv = nn.Conv2d(64 + 64, 3, kernel_size=(3, 3), padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, template):
        x = torch.cat([input, template], dim=1)
        down_128 = self.down0(x)
        down_64 = self.down1(down_128)
        down_32 = self.down2(down_64)
        down_16 = self.down3(down_32)
        down_8 = self.down4(down_16)
        up_8 = down_8
        up_16 = torch.cat([self.up1(up_8), down_16], dim=1)
        up_32 = torch.cat([self.up2(up_16), down_32], dim=1)
        up_64 = torch.cat([self.up3(up_32), down_64], dim=1)
        up_128 = torch.cat([self.up4(up_64), down_128], dim=1)
        img = self.conv(up_128)
        return img