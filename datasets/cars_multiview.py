from typing import List
import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
from os import listdir, path

import random
from collections import namedtuple
CarSequence = namedtuple('CarSequence', ['seq_id', 'frames', 'frames360', 'frame_frontal','rotation'])


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.car_sequences = get_multiview_metadata()
        random.seed(0)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.0), # TODO change to 0.5?
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.filename = 'tripod_seq_{:02d}_{:03d}.png'
        self.all_files = sorted(listdir(data_root))

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        cur_seq = int(img_path.split('_')[2])
        cur_frame = int(img_path.split('_')[3].split('.')[0])

        orig_img = Image.open(path.join(self.data_root, self.filename.format(cur_seq, cur_frame)))
        orig_img = orig_img.convert('RGB')
        img = self.transform(orig_img)

        frame_rotated, degrees = get_random_rotation(cur_seq, cur_frame, self.car_sequences)
        
        path_rotated = self.filename.format(cur_seq, frame_rotated)
        img_rotated = Image.open(path.join(self.data_root, path_rotated))
        img_rotated = img_rotated.convert('RGB')
        img_rotated = self.transform(img_rotated)

        # Add the same color jitter to both images
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3)
        transform = transforms.ColorJitter.get_params(
            color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
            color_jitter.hue)
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        order, brightness_factor, contrast_factor, saturation_factor, hue_factor = transform

        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)
        img = normalize((img))

        img_rotated = TF.adjust_brightness(img_rotated, brightness_factor)
        img_rotated = TF.adjust_contrast(img_rotated, contrast_factor)
        img_rotated = TF.adjust_saturation(img_rotated, saturation_factor)
        img_rotated = TF.adjust_hue(img_rotated, hue_factor)
        img_rotated = normalize((img_rotated))

        sample = {'img': img, 'img_rotated': img_rotated, 'degrees': degrees}
        return sample

    def __len__(self):
        return len(self.all_files)


class TrainRegSet(TrainSet):
    def __init__(self, data_root, image_size):
        super().__init__(data_root, image_size)


class TestSet(TrainSet):
    def __init__(self, data_root, image_size):
        super().__init__(data_root, image_size)


def regress_kp(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp*2)
    y = y.reshape(batch_size, n_gt_kp*2)
    try:
        beta = (X.T @ X).inverse() @ X.T @ y
    except:
        print('use penalty in linear regression')
        beta = (X.T @ X + 1e-3 * torch.eye(n_det_kp*2).to(X)).inverse() @ X.T @ y
    scaled_difference = (X @ beta - y).reshape(X.shape[0], n_gt_kp, 2)
    eval_acc = (scaled_difference.norm(dim=2) < 6).float().mean()
    return {'val_loss': -eval_acc, 'beta': beta}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    beta = regress_kp(valid_list)['beta']

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in test_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)
    scaled_difference = (X @ beta - y).reshape(X.shape[0], n_gt_kp, 2)
    eval_acc = (scaled_difference.norm(dim=2) < 6).float().mean()
    return {'val_loss': -eval_acc, 'beta': beta}


def get_multiview_metadata(path='cars/annotations/tripod-seq.txt') -> List:
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [' '.join(line.split()).split(' ') for line in lines]

    n_seq, width, height = [int(x) for x in lines[0]]
    car_sequences = []
    for i in range(0, n_seq):
        car_sequences.append(
            CarSequence(
                seq_id=i+1, 
                frames=int(lines[1][i]), 
                frames360=int(lines[4][i]), 
                frame_frontal=int(lines[5][i]), 
                rotation=int(lines[6][i]))
        )
    return car_sequences


def get_random_rotation(seq_id, frame, car_sequences):
    seq_id_index = seq_id - 1
    random_rotation_degrees = random.randint(0, 4) * 10 # TODO try larger rotations?
    total_frames = car_sequences[seq_id_index].frames
    frames_360 = car_sequences[seq_id_index].frames360
    direction = car_sequences[seq_id_index].rotation

    frames_per_degree = frames_360 / 360
    frame_rotated = ((frame + int(random_rotation_degrees * frames_per_degree)) % total_frames) + 1

    if direction == -1:
        random_rotation_degrees = -random_rotation_degrees

    return frame_rotated, random_rotation_degrees
