"""
06a_controllability_test.py
===========================
THE experiment that decides whether conditioning works. Loads the trained
conditional model, takes a few FIXED input canvases, and generates each one
9 times (every walkability x flood_resilience label combination). If the
model responds to labels, the outputs should CHANGE as the labels change.

Two outputs:
  1. controllability_grid.png  -- visual: rows = walkability low/med/high,
                                  cols = flood low/med/high, for one canvas.
                                  You should SEE the layout change across it.
  2. printed metrics + controllability_metrics.csv -- for each label setting,
     the measured green%, building-mix, and building% of the generated image.
     If "high walkability" produces higher mix than "low", conditioning works.

Runs in ~1 minute on GPU. No training.

Setup (same as training):
    from google.colab import drive; drive.mount('/content/drive')
    !cp /content/drive/MyDrive/dataset.zip . && unzip -q dataset.zip -d dataset
Set CKPT below to your model file, then run.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
CKPT      = "/content/drive/MyDrive/pix2pix_runs/conditional_final.pt"  # <- your model
DATA_DIR  = "dataset"
DRIVE_DIR = "/content/drive/MyDrive/pix2pix_runs"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_CANVASES = 3            # how many fixed test canvases to probe

# target palette RGB (must match script 04) -> for measuring output composition
GREEN=(0,170,0); WATER=(0,90,255); RESI=(255,190,0); COMM=(230,30,30)
INDU=(150,60,200); OTHER=(240,240,240); ROAD=(140,140,140)

# ---------------------------------------------------------------- model (must match 05)
def down(cin,cout,norm=True):
    l=[nn.Conv2d(cin,cout,4,2,1,bias=not norm)]
    if norm:l.append(nn.BatchNorm2d(cout))
    l.append(nn.LeakyReLU(0.2)); return nn.Sequential(*l)
def up(cin,cout,drop=False):
    l=[nn.ConvTranspose2d(cin,cout,4,2,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU()]
    if drop:l.insert(2,nn.Dropout(0.5))
    return nn.Sequential(*l)
class UNetGenerator(nn.Module):
    def __init__(self,cin=9,cout=3):
        super().__init__()
        self.d1=down(cin,64,norm=False);self.d2=down(64,128);self.d3=down(128,256)
        self.d4=down(256,512);self.d5=down(512,512);self.d6=down(512,512);self.d7=down(512,512)
        self.bottom=nn.Sequential(nn.Conv2d(512,512,4,2,1),nn.ReLU())
        self.u1=up(512,512,drop=True);self.u2=up(1024,512,drop=True);self.u3=up(1024,512,drop=True)
        self.u4=up(1024,512);self.u5=up(1024,256);self.u6=up(512,128);self.u7=up(256,64)
        self.final=nn.Sequential(nn.ConvTranspose2d(128,cout,4,2,1),nn.Tanh())
    def forward(self,x):
        d1=self.d1(x);d2=self.d2(d1);d3=self.d3(d2);d4=self.d4(d3)
        d5=self.d5(d4);d6=self.d6(d5);d7=self.d7(d6);b=self.bottom(d7)
        u=self.u1(b);u=self.u2(torch.cat([u,d7],1));u=self.u3(torch.cat([u,d6],1))
        u=self.u4(torch.cat([u,d5],1));u=self.u5(torch.cat([u,d4],1))
        u=self.u6(torch.cat([u,d3],1));u=self.u7(torch.cat([u,d2],1))
        return self.final(torch.cat([u,d1],1))

# ---------------------------------------------------------------- load model
G = UNetGenerator().to(DEVICE)
ck = torch.load(CKPT, map_location=DEVICE)
G.load_state_dict(ck["G"]); G.eval()
print(f"loaded {CKPT} (epoch {ck.get('epoch','?')})")

# ---------------------------------------------------------------- helpers
def load_input(tid):
    arr = np.asarray(Image.open(f"{DATA_DIR}/test/input/tile_{tid}.png").convert("RGB"),
                     dtype=np.float32)
    return torch.from_numpy(arr/127.5-1.0).permute(2,0,1).unsqueeze(0).to(DEVICE)

def gen(x, walk, flood):
    lab = torch.zeros(1,6,device=DEVICE)
    lab[0,walk]=1.0; lab[0,3+flood]=1.0
    maps = lab.view(1,6,1,1).expand(1,6,256,256)
    with torch.no_grad():
        out = G(torch.cat([x,maps],1))[0].cpu()
    return ((out+1)*127.5).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)

def nearest(px):
    """map each pixel to closest palette class."""
    cls = {"green":GREEN,"water":WATER,"resi":RESI,"comm":COMM,
           "indu":INDU,"other":OTHER,"road":ROAD,"bg":(0,0,0)}
    names=list(cls); cols=np.array([cls[n] for n in names])
    d = np.linalg.norm(px.reshape(-1,1,3).astype(int)-cols.reshape(1,-1,3),axis=2)
    idx = d.argmin(1)
    return {n:(idx==i).mean() for i,n in enumerate(names)}

def measure(img):
    f = nearest(img)
    build = f["resi"]+f["comm"]+f["indu"]+f["other"]
    # land-use mix (entropy) among building classes -> walkability proxy
    parts=np.array([f["resi"],f["comm"],f["indu"],f["other"]])
    p = parts/parts.sum() if parts.sum()>0 else parts
    ent = -(p[p>0]*np.log(p[p>0])).sum()/np.log(4) if parts.sum()>0 else 0.0
    return {"green_pct":f["green"], "building_pct":build,
            "mix_entropy":ent, "resi":f["resi"],"comm":f["comm"],"indu":f["indu"]}

# ---------------------------------------------------------------- pick canvases
man = pd.read_csv(f"{DATA_DIR}/manifest.csv")
test_ids = test_ids = [3268]
lvl=["low","med","high"]

# ---------------------------------------------------------------- run + measure
rows=[]
for tid in test_ids:
    x=load_input(tid)
    for w in range(3):
        for fl in range(3):
            m=measure(gen(x,w,fl))
            m.update({"tile":tid,"walk":lvl[w],"flood":lvl[fl]}); rows.append(m)
df=pd.DataFrame(rows)
df.to_csv("controllability_metrics.csv",index=False)

# ---- THE KEY NUMBERS: does mix rise with walkability? does green rise with flood?
print("\n=== CONTROLLABILITY (averaged over canvases) ===")
print("\nMix entropy by walkability label (expect: low < med < high if it works):")
print(df.groupby("walk")["mix_entropy"].mean().reindex(lvl).round(3).to_string())
print("\nGreen %% by flood label (expect: low < med < high if it works):")
print(df.groupby("flood")["green_pct"].mean().reindex(lvl).round(3).to_string())
print("\nBuilding %% by walkability (context):")
print(df.groupby("walk")["building_pct"].mean().reindex(lvl).round(3).to_string())

wm = df.groupby("walk")["mix_entropy"].mean().reindex(lvl)
gf = df.groupby("flood")["green_pct"].mean().reindex(lvl)
print("\n--- VERDICT ---")
print(f"walkability->mix monotonic increase: {wm.is_monotonic_increasing}  "
      f"(delta {wm['high']-wm['low']:+.3f})")
print(f"flood->green monotonic increase:     {gf.is_monotonic_increasing}  "
      f"(delta {gf['high']-gf['low']:+.3f})")
print("If deltas are clearly positive, conditioning works. If ~0, it's weak.")

# ---------------------------------------------------------------- visual grid (canvas 0)
tid=test_ids[0]; x=load_input(tid)
fig,axes=plt.subplots(3,3,figsize=(11,11))
for w in range(3):
    for fl in range(3):
        axes[w,fl].imshow(gen(x,w,fl))
        axes[w,fl].set_title(f"walk={lvl[w]}  flood={lvl[fl]}",fontsize=9)
        axes[w,fl].set_axis_off()
fig.suptitle(f"Same canvas (tile {tid}), all 9 label combinations",fontsize=13)
plt.tight_layout(); plt.savefig(f"{DRIVE_DIR}/controllability_grid.png",dpi=100)
plt.savefig("controllability_grid.png",dpi=100)
print(f"\nSaved controllability_grid.png (to Drive + local) and controllability_metrics.csv")
