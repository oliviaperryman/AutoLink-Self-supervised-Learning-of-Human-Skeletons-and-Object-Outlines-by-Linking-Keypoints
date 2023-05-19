import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

import numpy as np
from scipy.spatial.transform import Rotation as R


def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, 2) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y = torch.meshgrid([x, x], indexing='ij')
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid

def gen_grid3d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, grid_size, 3) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y, z = torch.meshgrid([x, x, x], indexing='ij')
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, grid_size, 3)
    return grid

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


class Detector(nn.Module):
    def __init__(self, hyper_paras: pl.utilities.parsing.AttributeDict) -> None:
        super().__init__()
        self.n_parts = hyper_paras.n_parts
        self.output_size = 16

        self.conv = nn.Sequential(
            ResBlock(3, 64),  # 64
            ResBlock(64, 128),  # 32
            ResBlock(128, 256),  # 16
            ResBlock(256, 512),  # 8
            TransposedBlock(512, 256),  # 16
            # save memory in 3D, make grid only 16x16x16
            # TransposedBlock(256, 128),  # 32 
            nn.Conv2d(256, self.n_parts * self.output_size, kernel_size=3, padding=1),
            torch.nn.Unflatten(1, (self.n_parts,self.output_size)),
        )

        # grid = gen_grid2d(self.output_size).reshape(1, 1, self.output_size ** 2, 2) 
        grid = gen_grid3d(self.output_size).reshape(1, 1, self.output_size ** 3, 3)
        self.coord = nn.Parameter(grid, requires_grad=False)

    def forward(self, input_dict: dict) -> dict:
        img = F.interpolate(input_dict['img'], size=(128, 128), mode='bilinear', align_corners=False)
        prob_map = self.conv(img).reshape(img.shape[0], self.n_parts, -1, 1)
        prob_map = F.softmax(prob_map, dim=2)
        keypoints = self.coord * prob_map
        keypoints = keypoints.sum(dim=2)
        prob_map = prob_map.reshape(keypoints.shape[0], self.n_parts, self.output_size, self.output_size, self.output_size)

        # rotate keypoints
        # center keypoints around 0,0,0
        # centered_keypoints = keypoints - self.output_size / 2
        
        # Change keypoints from (depth, height, width) to (width, height, depth)
        perm = torch.LongTensor([2, 1, 0])
        keypoints_xyz = keypoints[:, :, perm]

        detached_keypoints = keypoints_xyz.detach().cpu().numpy()

        # rotate around y axis
        for batch in range(keypoints.shape[0]):
            r = R.from_rotvec(input_dict["degrees"][batch].detach().cpu().numpy() * np.array([0, 0, 1]), degrees=True)
            for kp in range(keypoints.shape[1]):
                detached_keypoints[batch, kp] = r.apply(detached_keypoints[batch, kp])

        keypoints_xyz_rotated = torch.tensor(detached_keypoints).to(input_dict['img'].device)

        # move keypoints back to original position
        # keypoints = centered_keypoints + self.output_size / 2

        # Change keypoints from (width, height, depth) to (width, height)
        keypoints_xy_rotated = keypoints_xyz_rotated[:,:,:2]

        # Change keypoints back to (height, width)
        perm = torch.LongTensor([1, 0])
        keypoints = keypoints_xy_rotated[:, :, perm]

        return {'keypoints': keypoints, 'prob_map': prob_map}


class Encoder(nn.Module):
    def __init__(self, hyper_paras: pl.utilities.parsing.AttributeDict) -> None:
        super().__init__()
        self.detector = Detector(hyper_paras)
        self.missing = hyper_paras.missing
        self.block = hyper_paras.block

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_dict: dict, need_masked_img: bool=False) -> dict:
        mask_batch = self.detector(input_dict)
        if need_masked_img:
            damage_mask = torch.zeros(input_dict['img_rotated'].shape[0], 1, self.block, self.block, device=input_dict['img_rotated'].device).uniform_() > self.missing
            damage_mask = F.interpolate(damage_mask.to(input_dict['img_rotated']), size=input_dict['img_rotated'].shape[-1], mode='nearest')
            mask_batch['damaged_img'] = input_dict['img_rotated'] * damage_mask
        return mask_batch
