from collections import defaultdict
import json
import os

import numpy as np
import scipy.io
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.camera_parameters = json.load(open("/local/omp/human36m-camera-parameters/camera-parameters.json","r"))

        self.cameras = ["54138969", "55011271", "58860488", "60457274"]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.to_tensor = transforms.ToTensor()

        self.data_root = data_root

        self.samples = []

        actions = ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']
        grouped_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for folder_name in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                if folder_name.startswith(tuple(actions)):
                    action_name, camera_name = folder_name.split('.')
                    for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                'WithBackground', folder_name)):
                        grouped_actions[subject_index][action_name][frame_index.split('.')[0]].append((subject_index, folder_name, frame_index.split('.')[0], action_name, camera_name))
        
        for s in grouped_actions.keys():
            for a in grouped_actions[s].keys():
                for f in grouped_actions[s][a].keys():
                    self.samples.append(grouped_actions[s][a][f])

    def __getitem__(self, idx):

        multiviews = {}
        # 4 views for each sample
        for view in self.samples[idx]:
            subject_index, folder_names, frame_index, action_name, camera = view
            img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                        folder_names, '{}.jpg'.format(frame_index)))
            mask = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'BackgroudMask',
                                        folder_names, '{}.png'.format(frame_index)))
            
            img_wobg = self.transform(img) * self.to_tensor(mask)
            multiviews[camera] = img_wobg

        # projection from one view to another
        P1 = self.proj_matrix(subject_index, self.cameras[0])
        P2 = self.proj_matrix(subject_index, self.cameras[1])

        # rotation_matrix = P1 @ np.linalg.inv(P2)
        rotation_matrix = P1

        return {'img': multiviews[self.cameras[0]], 'img_rotated': multiviews[self.cameras[1]], 'rotation_matrix': rotation_matrix }

    def __len__(self):
        return len(self.samples)
    
    def proj_matrix(self, subject_index, camera):
        calibration_matrix = self.camera_parameters["intrinsics"][camera]["calibration_matrix"]
        R =  self.camera_parameters["extrinsics"][f"S{subject_index}"][camera]["R"]
        t =  self.camera_parameters["extrinsics"][f"S{subject_index}"][camera]["t"]

        P = calibration_matrix @ np.hstack([R, t])
        return P


class TrainRegSet(TrainSet):
    def __init__(self, data_root, image_size):
        super().__init__()
    

class TestSet(TrainSet):
    def __init__(self, data_root, image_size):
        super().__init__()
        

correspondences = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (17, 25), (18, 26), (19, 27), (20, 28), (21, 28), (22, 30), (23, 31)]

def swap_points(points):
    """
    points: B x N x D
    """
    permutation = list(range((points.shape[1])))
    for a, b in correspondences:
        permutation[a] = b
        permutation[b] = a
    new_points = points[:, permutation, :].clone()
    return new_points


def regress_kp(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)
    XTXXT = (X.T @ X).inverse() @ X.T

    while True:
        beta = XTXXT @ y
        pred_y = X @ beta

        dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        swaped_y = swap_points(y.reshape(batch_size, n_gt_kp, 2)).reshape(batch_size, n_gt_kp * 2)
        swaped_dist = (pred_y - swaped_y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        should_swap = dist > swaped_dist

        if should_swap.sum() > 10:
            y[should_swap] = swaped_y[should_swap].clone()
        else:
            break

    dist_mean = dist.mean()
    dist_std = dist.std()
    chosen = dist < dist_mean + 3 * dist_std
    X, y = X[chosen], y[chosen]

    beta = (X.T @ X).inverse() @ X.T @ y
    pred_y = X @ beta
    dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

    return {'val_loss': dist.mean(), 'beta': beta}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    beta = regress_kp(valid_list)['beta']

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in test_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)

    pred_y = X @ beta

    while True:
        dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)
        swaped_y = swap_points(y.reshape(batch_size, n_gt_kp, 2)).reshape(batch_size, n_gt_kp * 2)
        swaped_dist = (pred_y - swaped_y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        should_swap = dist > swaped_dist

        if should_swap.sum() > 10:
            y[should_swap] = swaped_y[should_swap].clone()
        else:
            break

    return {'val_loss': dist.mean(), 'beta': beta}
