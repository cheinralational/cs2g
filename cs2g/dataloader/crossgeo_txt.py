import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from torchvision import transforms


class CrossGEODataset(Dataset):
    def __init__(self, data_root="./dataset/crossgeo/data/", sat_size=(256, 256), pano_size=(256, 256), split="train", include_uav=False):
        self.data_root = data_root
        self.sat_size = sat_size
        self.pano_size = pano_size
        self.split = split
        self.include_uav = include_uav

        pairs = sorted([d for d in os.listdir(data_root) if d.startswith("pair_") and os.path.isdir(os.path.join(data_root, d))])

        all_samples = []
        for pair in pairs:
            pair_dir = os.path.join(data_root, pair)
            quad_info_path = os.path.join(pair_dir, "quad_info.json")
            if os.path.exists(quad_info_path):
                with open(quad_info_path, 'r') as f:
                    quad_info = json.load(f)
            else:
                quad_info = {}

            for g_idx in [1, 2]:
                ground_rgb = os.path.join(pair_dir, f"ground_{g_idx}_rgb.jpg")
                satellite_img = os.path.join(pair_dir, f"ground_{g_idx}_satellite.png")
                if os.path.exists(ground_rgb):
                    all_samples.append({
                        'sat': satellite_img,
                        'pano': ground_rgb,
                        'pair': pair,
                        'ground_idx': g_idx,
                        'quad_info': quad_info,
                    })

            if include_uav:
                for u_idx in [1, 2]:
                    uav_rgb = os.path.join(pair_dir, f"uav_{u_idx}_rgb.jpg")
                    satellite_img = os.path.join(pair_dir, f"uav_{u_idx}_rgb.jpg")
                    if os.path.exists(uav_rgb):
                        pass

        n_total = len(all_samples)
        n_train = int(n_total * 0.8)

        if split in ("train", "training"):
            self.samples = all_samples[:n_train]
        elif split in ("val", "test", "validation"):
            self.samples = all_samples[n_train:]
        else:
            self.samples = all_samples

        self.sat_transform = transforms.Compose([
            transforms.Resize(sat_size),
            transforms.ToTensor(),
        ])
        self.pano_transform = transforms.Compose([
            transforms.Resize(pano_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sat_img = Image.open(sample['sat']).convert('RGB')
        pano_img = Image.open(sample['pano']).convert('RGB')

        sat_tensor = self.sat_transform(sat_img)
        pano_tensor = self.pano_transform(pano_img)

        quad_info = sample.get('quad_info', {})

        return {
            'sat': sat_tensor,
            'pano': pano_tensor,
            'paths': sample['pano'],
            'label': torch.tensor(idx, dtype=torch.long),
            'shift_x': quad_info.get('shift_x', 0.0),
            'shift_y': quad_info.get('shift_y', 0.0),
            'intrinsics': quad_info.get('intrinsics', [1.0, 1.0, 1.0, 1.0]),
            'c2w': quad_info.get('c2w', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        }
