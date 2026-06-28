import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cs2g'))
from utils.util import instantiate_from_config
import yaml

print('=== Testing Stage1 model instantiation ===')
with open(r'd:\小研究\zxm\cs2g\configs\Boost_Sat2Den\crossgeo_geo_ldm_stage1.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
model = instantiate_from_config(cfg['model'])
print(f'OK: Stage1 model, type={type(model).__name__}')
print(f'  use_spherical_control={model.use_spherical_control}')
print(f'  DDPM type={type(model.DDPM).__name__}')

print()
print('=== Testing Stage2 model instantiation ===')
with open(r'd:\小研究\zxm\cs2g\configs\Boost_Sat2Den\crossgeo_geo_ldm_sphere.yaml', 'r') as f:
    cfg2 = yaml.safe_load(f)
model2 = instantiate_from_config(cfg2['model'])
print(f'OK: Stage2 model, type={type(model2).__name__}')
print(f'  use_spherical_control={model2.use_spherical_control}')
print(f'  spherical_controlnet={model2.DDPM.spherical_controlnet is not None}')

print()
print('=== Testing dataloader ===')
from dataloader.crossgeo_txt import CrossGEODataset
ds = CrossGEODataset(data_root='./dataset/crossgeo/data/', split='train')
print(f'OK: Train dataset, samples={len(ds)}')
sample = ds[0]
print(f'  sat shape={sample["sat"].shape}, pano shape={sample["pano"].shape}')
print(f'  keys={list(sample.keys())}')
print()
print('ALL TESTS PASSED!')
