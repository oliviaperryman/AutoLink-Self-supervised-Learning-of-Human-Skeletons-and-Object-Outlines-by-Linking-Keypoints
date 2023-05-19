import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, 2) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y = torch.meshgrid([x, x], indexing='xy')
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
        self.output_size = 32

        self.conv = nn.Sequential(
            ResBlock(3, 64),  # 64
            ResBlock(64, 128),  # 32
            ResBlock(128, 256),  # 16
            ResBlock(256, 512),  # 8
            TransposedBlock(512, 256),  # 16
            TransposedBlock(256, 128),  # 32 
            nn.Conv2d(128, self.n_parts * 2, kernel_size=3, padding=1), # double channels for xy and depth
        )

        grid = gen_grid2d(self.output_size).reshape(1, 1, self.output_size ** 2, 2) 
        self.coord = nn.Parameter(grid, requires_grad=False)

    def forward(self, input_dict: dict) -> dict:
        img = F.interpolate(input_dict['img'], size=(128, 128), mode='bilinear', align_corners=False)
        prob_maps = self.conv(img)
        prob_map_xy = prob_maps[:, :self.n_parts, :, :]
        prob_map_z = prob_maps[:, self.n_parts:, :, :]

        # calculate keypoints
        prob_map_xy = prob_map_xy.reshape(img.shape[0], self.n_parts, -1, 1)
        prob_map_xy = F.softmax(prob_map_xy, dim=2)
        keypoints = self.coord * prob_map_xy
        keypoints = keypoints.sum(dim=2)

        # calculate depth
        prob_map_z = prob_map_z.reshape(img.shape[0], self.n_parts, -1, 1)
        z = prob_map_z * prob_map_xy
        z = z.sum(dim=2)
        # normalize to -1, 1
        z = torch.tanh(z)
        # depth = torch.tanh(z / 10) if depth gets too big

        keypoints_xyz = torch.cat((keypoints,z), dim=-1)

        # rotate keypoints
        r = R.from_rotvec([deg * np.array([0, 1, 0]) for deg in input_dict["degrees"].detach().cpu().numpy()], degrees=True)
        matrix = r.as_matrix()
        keypoints_xyz_rot = torch.einsum('ijk,imk->imj', torch.tensor(matrix).to('cuda').float(), keypoints_xyz.clone())

        prob_map_xy = prob_map_xy.reshape(keypoints.shape[0], self.n_parts, self.output_size, self.output_size)

        return {'keypoints': keypoints_xyz_rot, 'prob_map': prob_map_xy, "keypoints_before_rotation": keypoints_xyz}

# Debugging functions
def save_2D_kp(kp, name):
    plt.clf()
    plt.scatter(kp[0].detach().cpu()[:,0], kp[0].detach().cpu()[:,1])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig(name)

def save_2D_kp_rot(kp, kp_rot, name):
    plt.clf()
    plt.scatter(kp[0].detach().cpu()[:,0], kp[0].detach().cpu()[:,1])
    plt.scatter(kp_rot[0].detach().cpu()[:,0], kp_rot[0].detach().cpu()[:,1])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig(name)

def save_3D_kp(kp, name, elev=90, azim=270, roll=0):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(kp[0].detach().cpu()[:,0], kp[0].detach().cpu()[:,1], kp[0].detach().cpu()[:,2])
    ax.view_init(elev=elev, azim=azim, roll=roll)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig(name)


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
