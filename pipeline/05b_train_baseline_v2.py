"""
05b_train_baseline_v2.py
========================
BASELINE RETRAIN with weakened discriminator (optD lr = LR/4).
The first baseline collapsed at epoch 1 (D won instantly and permanently);
cutting the discriminator's learning rate 4x gives the label-blind generator
room to compete.

BEFORE RUNNING — rename the collapsed run's outputs so they're preserved:
    import os
    D="/content/drive/MyDrive/pix2pix_runs/"
    for f in ["baseline_final.pt","baseline_losses.csv","baseline_samples_final.png"]:
        if os.path.exists(D+f): os.rename(D+f, D+"collapsed_"+f)
    if os.path.exists(D+"baseline_latest.pt"): os.remove(D+"baseline_latest.pt")
(removing baseline_latest.pt matters — otherwise this run RESUMES the collapsed state!)

WATCH THE FIRST 20 EPOCHS: if D holds above ~0.05 instead of flatlining at
0.000, the fix took. If it collapses again, let it finish anyway (script 06's
zero-label ablation is the fallback comparison).
"""


import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------- settings
BASELINE   = True           # BASELINE RETRAIN SCRIPT: leave True
EPOCHS     = 200
BATCH      = 16
LR         = 2e-4
L1_WEIGHT  = 100.0
SAVE_EVERY = 50             # rare checkpoint, single overwritten file
DATA_DIR   = "dataset"
DRIVE_DIR  = "/content/drive/MyDrive/pix2pix_runs"
RUN_NAME   = "baseline" if BASELINE else "conditional"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "No GPU! Runtime > Change runtime type > T4 GPU"
os.makedirs(DRIVE_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
LATEST = f"{DRIVE_DIR}/{RUN_NAME}_latest.pt"

# ---------------------------------------------------------------- dataset
class TileDataset(Dataset):
    def __init__(self, split):
        man = pd.read_csv(f"{DATA_DIR}/manifest.csv")
        self.rows = man[man["split"] == split].reset_index(drop=True)
        self.split = split
    def __len__(self): return len(self.rows)
    def _load(self, path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1)
    def __getitem__(self, i):
        r = self.rows.iloc[i]; tid = int(r["tile_id"])
        x = self._load(f"{DATA_DIR}/{self.split}/input/tile_{tid}.png")
        y = self._load(f"{DATA_DIR}/{self.split}/target/tile_{tid}.png")
        lab = torch.zeros(6)
        lab[int(r["walk_lab"])] = 1.0
        lab[3 + int(r["flood_lab"])] = 1.0
        if BASELINE: lab = torch.zeros(6)
        if self.split == "train":
            k = np.random.randint(4)
            if k: x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if np.random.rand() < 0.5:
                x, y = torch.flip(x, (2,)), torch.flip(y, (2,))
        return x, y, lab, tid

def with_labels(img, lab):
    B, _, H, W = img.shape
    return torch.cat([img, lab.view(B, 6, 1, 1).expand(B, 6, H, W)], dim=1)

# ---------------------------------------------------------------- models
def down(cin, cout, norm=True):
    l = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm: l.append(nn.BatchNorm2d(cout))
    l.append(nn.LeakyReLU(0.2)); return nn.Sequential(*l)

def up(cin, cout, drop=False):
    l = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
         nn.BatchNorm2d(cout), nn.ReLU()]
    if drop: l.insert(2, nn.Dropout(0.5))
    return nn.Sequential(*l)

class UNetGenerator(nn.Module):
    def __init__(self, cin=9, cout=3):
        super().__init__()
        self.d1 = down(cin, 64, norm=False); self.d2 = down(64, 128)
        self.d3 = down(128, 256); self.d4 = down(256, 512)
        self.d5 = down(512, 512); self.d6 = down(512, 512); self.d7 = down(512, 512)
        self.bottom = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.u1 = up(512, 512, drop=True);  self.u2 = up(1024, 512, drop=True)
        self.u3 = up(1024, 512, drop=True); self.u4 = up(1024, 512)
        self.u5 = up(1024, 256); self.u6 = up(512, 128); self.u7 = up(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, cout, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1=self.d1(x); d2=self.d2(d1); d3=self.d3(d2); d4=self.d4(d3)
        d5=self.d5(d4); d6=self.d6(d5); d7=self.d7(d6); b=self.bottom(d7)
        u=self.u1(b); u=self.u2(torch.cat([u,d7],1)); u=self.u3(torch.cat([u,d6],1))
        u=self.u4(torch.cat([u,d5],1)); u=self.u5(torch.cat([u,d4],1))
        u=self.u6(torch.cat([u,d3],1)); u=self.u7(torch.cat([u,d2],1))
        return self.final(torch.cat([u,d1],1))

class PatchDiscriminator(nn.Module):
    def __init__(self, cin=9+3):
        super().__init__()
        self.net = nn.Sequential(
            down(cin, 64, norm=False), down(64, 128), down(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1))
    def forward(self, xl, y): return self.net(torch.cat([xl, y], 1))

# ---------------------------------------------------------------- setup
train_ds, test_ds = TileDataset("train"), TileDataset("test")
train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2, drop_last=True)
print(f"run={RUN_NAME} | train {len(train_ds)} | test {len(test_ds)} | {DEVICE}")

G = UNetGenerator().to(DEVICE)
D = PatchDiscriminator().to(DEVICE)
optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=LR/4, betas=(0.5, 0.999))  # weakened D: fixes ep-1 collapse
bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

fix_x, fix_y, fix_l, fix_ids = next(iter(DataLoader(test_ds, 6, shuffle=False)))
fix_x, fix_l = fix_x.to(DEVICE), fix_l.to(DEVICE)

# ---- resume if a latest-checkpoint exists --------------------------------
start_epoch, history = 0, []
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE)
    G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
    optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
    start_epoch, history = ck["epoch"] + 1, ck["history"]
    print(f"RESUMED from epoch {ck['epoch']}")

# ---------------------------------------------------------------- training
print("training...")
for epoch in range(start_epoch, EPOCHS):
    t0, gl, dl, n = time.time(), 0.0, 0.0, 0
    for x, y, lab, _ in train_dl:
        x, y, lab = x.to(DEVICE), y.to(DEVICE), lab.to(DEVICE)
        xl = with_labels(x, lab)
        with torch.no_grad(): fake = G(xl)
        d_r, d_f = D(xl, y), D(xl, fake)
        lossD = 0.5*(bce(d_r, torch.ones_like(d_r)) + bce(d_f, torch.zeros_like(d_f)))
        optD.zero_grad(); lossD.backward(); optD.step()
        fake = G(xl); d_o = D(xl, fake)
        lossG = bce(d_o, torch.ones_like(d_o)) + L1_WEIGHT*l1(fake, y)
        optG.zero_grad(); lossG.backward(); optG.step()
        gl += lossG.item(); dl += lossD.item(); n += 1

    history.append({"epoch": epoch, "G": gl/n, "D": dl/n})
    print(f"ep {epoch:3d} | G {gl/n:6.3f} | D {dl/n:5.3f} | {time.time()-t0:4.0f}s")

    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "epoch": epoch, "history": history}, LATEST)   # OVERWRITES
        pd.DataFrame(history).to_csv(f"{DRIVE_DIR}/{RUN_NAME}_losses.csv", index=False)
        print(f"  checkpoint overwritten ({RUN_NAME}_latest.pt, ep {epoch})")

# ---------------------------------------------------------------- final save
torch.save({"G": G.state_dict(), "epoch": EPOCHS-1, "history": history},
           f"{DRIVE_DIR}/{RUN_NAME}_final.pt")
pd.DataFrame(history).to_csv(f"{DRIVE_DIR}/{RUN_NAME}_losses.csv", index=False)
G.eval()
with torch.no_grad():
    fake = G(with_labels(fix_x, fix_l)).cpu()
grid = ((torch.cat([fix_x.cpu(), fake], 0) + 1) * 127.5).clamp(0, 255)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for j in range(6):
    axes[0, j].imshow(grid[j].permute(1, 2, 0).numpy().astype(np.uint8))
    axes[1, j].imshow(grid[6+j].permute(1, 2, 0).numpy().astype(np.uint8))
    axes[0, j].set_title(f"in {int(fix_ids[j])}", fontsize=8)
for ax in axes.flat: ax.set_axis_off()
plt.tight_layout()
plt.savefig(f"{DRIVE_DIR}/{RUN_NAME}_samples_final.png", dpi=90)
print(f"DONE: {RUN_NAME}_final.pt, {RUN_NAME}_losses.csv, "
      f"{RUN_NAME}_samples_final.png on Drive. You may delete {RUN_NAME}_latest.pt.")
