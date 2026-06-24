import sys, os
sys.path.insert(0, r'd:\小研究\zxm')

data_root = r'd:\小研究\zxm\dataset\crossgeo\data'

pair_dirs = sorted([
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d)) and d.startswith('pair_')
])
print(f'Pairs found: {len(pair_dirs)} -> {pair_dirs}')

all_samples = []
for pair_dir in pair_dirs:
    pair_path = os.path.join(data_root, pair_dir)
    for g_idx in range(1, 3):
        grd = os.path.join(pair_path, f'ground_{g_idx}_rgb.jpg')
        sat_jpg = os.path.join(pair_path, f'ground_{g_idx}_satellite.jpg')
        sat_png = os.path.join(pair_path, f'ground_{g_idx}_satellite.png')
        sat = sat_jpg if os.path.exists(sat_jpg) else sat_png
        npy = os.path.join(pair_path, f'ground_{g_idx}_rgb.npy')
        depth = os.path.join(pair_path, f'ground_{g_idx}_depth.npy')
        ok = os.path.exists(grd) and os.path.exists(sat)
        print(f'  {pair_dir}/ground_{g_idx}: rgb={os.path.exists(grd)}, sat={os.path.exists(sat)}, npy={os.path.exists(npy)}, depth={os.path.exists(depth)} => {"OK" if ok else "MISSING"}')
        if ok:
            all_samples.append({'grd_path': grd, 'sat_path': sat, 'npy_path': npy, 'depth_path': depth, 'pair': pair_dir, 'view': f'ground_{g_idx}'})

    for u_idx in range(1, 3):
        uav = os.path.join(pair_path, f'uav_{u_idx}_rgb.jpg')
        sat_jpg = os.path.join(pair_path, 'ground_1_satellite.jpg')
        sat_png = os.path.join(pair_path, 'ground_1_satellite.png')
        sat = sat_jpg if os.path.exists(sat_jpg) else sat_png
        npy = os.path.join(pair_path, f'uav_{u_idx}_rgb.npy')
        depth = os.path.join(pair_path, f'uav_{u_idx}_depth.npy')
        ok = os.path.exists(uav) and os.path.exists(sat)
        print(f'  {pair_dir}/uav_{u_idx}: rgb={os.path.exists(uav)}, sat={os.path.exists(sat)}, npy={os.path.exists(npy)}, depth={os.path.exists(depth)} => {"OK" if ok else "MISSING"}')
        if ok:
            all_samples.append({'grd_path': uav, 'sat_path': sat, 'npy_path': npy, 'depth_path': depth, 'pair': pair_dir, 'view': f'uav_{u_idx}'})

print(f'\nGROUND only: {sum(1 for s in all_samples if s["view"].startswith("ground"))} samples')
print(f'UAV only: {sum(1 for s in all_samples if s["view"].startswith("uav"))} samples')
print(f'TOTAL: {len(all_samples)} samples')

import numpy as np
npy_meta = np.load(all_samples[0]['npy_path'], allow_pickle=True).item()
print(f'\nSample npy keys: {list(npy_meta.keys())}')
print(f'  intrinsics: {npy_meta["intrinsics"].shape} {npy_meta["intrinsics"].dtype}')
print(f'  c2w: {npy_meta["c2w"].shape} {npy_meta["c2w"].dtype}')
print(f'  raw_data: {npy_meta["raw_data"]}')

from PIL import Image
im = Image.open(all_samples[0]['grd_path'])
print(f'\nSample image: {im.size} {im.mode}')
im2 = Image.open(all_samples[0]['sat_path'])
print(f'Sample satellite: {im2.size} {im2.mode}')
