"""
06b_fix_baseline_eval.py
========================
Fixes two evaluation-design issues found after the first 06 run:

1. BASELINE was evaluated with one-hot labels it never saw in training
   (its label-channel weights are untrained), producing garbage numbers.
   -> re-evaluate it in its NATIVE mode: zero labels. Its outputs are then
   label-invariant BY CONSTRUCTION (deltas exactly 0), and its FID becomes
   a fair realism measure.

2. The zero-label ABLATION of the conditional model is itself slightly
   out-of-distribution (the model always saw one-hot labels in training).
   -> add a SHUFFLED-LABEL ablation: valid one-hot labels, randomly wrong.
   In-distribution inputs carrying zero information. FID(true) vs
   FID(shuffled) cleanly isolates what correct label information contributes.

Appends corrected numbers to RESULTS_SUMMARY.txt on Drive.
Setup identical to 06 (mount, unzip dataset, pip install pytorch-fid).
Runtime ~5 min.
"""

import os, shutil, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image

DRIVE="/content/drive/MyDrive/pix2pix_runs"; OUT=f"{DRIVE}/evaluation"
DATA_DIR="dataset"; DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LVL=["low","med","high"]
np.random.seed(42)

# ---- model def (same as 06) ---------------------------------------------
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

def load_G(p):
    G=UNetGenerator().to(DEVICE); ck=torch.load(p,map_location=DEVICE)
    G.load_state_dict(ck["G"]); G.eval(); return G

Gc=load_G(f"{DRIVE}/conditional_final.pt")
Gb=load_G(f"{DRIVE}/baseline_final.pt")
man=pd.read_csv(f"{DATA_DIR}/manifest.csv")
test=man[man.split=="test"].reset_index(drop=True)
print(f"test canvases: {len(test)}")

def load_input(tid):
    a=np.asarray(Image.open(f"{DATA_DIR}/test/input/tile_{tid}.png").convert("RGB"),
                 dtype=np.float32)
    return torch.from_numpy(a/127.5-1.0).permute(2,0,1)
def to_img(t):
    return ((t+1)*127.5).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
def gen(G,xs,walks=None,floods=None):
    B=xs.shape[0]; lab=torch.zeros(B,6)
    if walks is not None:
        for i,(w,f) in enumerate(zip(walks,floods)):
            lab[i,w]=1.0; lab[i,3+f]=1.0
    maps=lab.view(B,6,1,1).expand(B,6,256,256)
    with torch.no_grad():
        return G(torch.cat([xs,maps],1).to(DEVICE)).cpu()

# ---- regenerate baseline (zero labels) + shuffled-label conditional ------
dirs={"base0":f"{OUT}/fid_base_zero","shuf":f"{OUT}/fid_cond_shuffled"}
for d in dirs.values():
    shutil.rmtree(d,ignore_errors=True); os.makedirs(d)

true_w=[int(r.walk_lab) for _,r in test.iterrows()]
true_f=[int(r.flood_lab) for _,r in test.iterrows()]
perm=np.random.permutation(len(test))          # shuffled label assignment
shuf_w=[true_w[i] for i in perm]; shuf_f=[true_f[i] for i in perm]

BS=16
for s in range(0,len(test),BS):
    chunk=test.iloc[s:s+BS]
    xs=torch.stack([load_input(int(t)) for t in chunk.tile_id])
    ob=gen(Gb,xs)                                             # native zero-label
    osf=gen(Gc,xs,shuf_w[s:s+len(chunk)],shuf_f[s:s+len(chunk)])
    for j,(_,r) in enumerate(chunk.iterrows()):
        tid=int(r.tile_id)
        Image.fromarray(to_img(ob[j])).save(f"{dirs['base0']}/{tid}.png")
        Image.fromarray(to_img(osf[j])).save(f"{dirs['shuf']}/{tid}.png")
print("regenerated baseline (zero-label) + shuffled-label conditional")

def fid(a,b):
    r=subprocess.run(["python","-m","pytorch_fid",a,b,"--device",DEVICE],
                     capture_output=True,text=True)
    for line in r.stdout.splitlines():
        if "FID" in line: return float(line.split()[-1])
    raise RuntimeError(r.stdout+r.stderr)

real=f"{OUT}/fid_real"
f_base=fid(real,dirs["base0"]); f_shuf=fid(real,dirs["shuf"])
s=("\n=== CORRECTED (06b) ===\n"
   f"FID real vs baseline (native zero-label mode): {f_base:.2f}\n"
   f"FID real vs conditional w/ SHUFFLED labels:    {f_shuf:.2f}\n"
   "baseline label-response deltas: 0 by construction (label channels untrained)\n"
   "NOTE: previous baseline sweep/FID numbers (one-hot inputs) are invalid —\n"
   "they probed untrained label-channel weights out of distribution.\n")
print(s)
with open(f"{OUT}/RESULTS_SUMMARY.txt","a") as fh: fh.write(s)
print("appended to RESULTS_SUMMARY.txt")

# ---- per-tile L1 reconstruction error, true vs shuffled labels (run after the code above)
l1t, l1s = [], []
for s in range(0, len(test), 16):
    ch = test.iloc[s:s+16]
    xs = torch.stack([load_input(int(t)) for t in ch.tile_id])
    ys = torch.stack([torch.from_numpy(np.asarray(Image.open(
         f"dataset/test/target/tile_{int(t)}.png").convert("RGB"),
         dtype=np.float32)/127.5-1.0).permute(2,0,1) for t in ch.tile_id])
    ot = gen(Gc, xs, true_w[s:s+len(ch)], true_f[s:s+len(ch)])
    osf = gen(Gc, xs, shuf_w[s:s+len(ch)], shuf_f[s:s+len(ch)])
    l1t.append((ot-ys).abs().mean().item()); l1s.append((osf-ys).abs().mean().item())
print(f"per-tile L1 vs real: true labels {np.mean(l1t):.4f}  shuffled {np.mean(l1s):.4f}")
