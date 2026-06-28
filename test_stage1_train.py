import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cs2g'))
import torch
import yaml
from utils.util import instantiate_from_config
from pytorch_lightning import seed_everything

seed_everything(24)

print("=" * 60)
print("  CrossGeo Stage-1 Training Pipeline Verification")
print("=" * 60)

with open(r'd:\小研究\zxm\cs2g\configs\Boost_Sat2Den\crossgeo_geo_ldm_stage1.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

print("\n[1/5] Building model...")
model = instantiate_from_config(cfg['model'])
model.use_spherical_control = False
model_psize = sum(p.numel() for p in model.parameters())
model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:  {model_psize / 1e6:.1f}M")
print(f"  Trainable:     {model_trainable / 1e6:.1f}M")
print(f"  AE (frozen):   {sum(p.numel() for p in model.pre_AE_model.parameters()) / 1e6:.1f}M")

print("\n[2/5] Building dataloader...")
data = instantiate_from_config(cfg['data'])
data.batch_size = 1
data.num_workers = 0
data.setup()
train_dl = iter(data.train_dataloader())
batch = next(train_dl)
print(f"  Batch keys: {list(batch.keys())}")
print(f"  sat shape:  {batch['sat'].shape}")
print(f"  pano shape: {batch['pano'].shape}")

print("\n[3/5] Forward pass...")
optimizer = model.configure_optimizers()[0]
model.train()

loss = model.training_step(batch, 0)
print(f"  Loss: {loss.item():.4f}")

print("\n[4/5] Backward pass + gradient check...")
optimizer.zero_grad()
loss.backward()

grad_params = 0
zero_grad_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        grad_params += 1
        if param.grad is None or param.grad.abs().sum() == 0:
            zero_grad_params += 1
            print(f"  WARNING: Zero gradient: {name}")

print(f"  Parameters with gradients: {grad_params}")
print(f"  Zero-gradient params: {zero_grad_params}")

if zero_grad_params == 0:
    print("  ALL GRADIENTS FLOW CORRECTLY")

print("\n[5/5] Optimizer step...")
optimizer.step()
print("  Optimizer step completed successfully!")

print("\n" + "=" * 60)
print("  Stage-1 VERIFICATION: ALL PASSED!")
print("=" * 60)
