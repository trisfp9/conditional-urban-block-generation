"""
06_evaluate.py
==============
The full evaluation. Produces every number and figure for the Results section.

Experiments:
  1. REALISM        — FID between generated layouts (true labels, held-out
                      canvases) and the real held-out layouts, + for the
                      zero-label ablation and (if present) the retrained
                      baseline.
  2. CONTROLLABILITY— for EVERY test canvas, generate all 9 label combos and
                      measure green%, mix entropy, building%. Deltas across
                      labels = strength of conditioning.
  3. TRADEOFF MAP   — 3x3 heatmaps (mean green / mix / building per combo):
                      which combinations are achievable, and emergent density
                      across the space.
  4. COMPARISON     — the same 9-sweep run through the baseline (if its
                      retrain worked): its deltas should be ~0. Plus the
                      zero-label ABLATION of the conditional model (labels
                      zeroed at eval), compared on FID/L1 — isolates what the
                      labels contribute. Ablation runs regardless of baseline.

Everything (CSVs, figures, summary txt) saves DIRECTLY TO DRIVE.

Setup:
    from google.colab import drive; drive.mount('/content/drive')
    !cp /content/drive/MyDrive/dataset.zip . && unzip -q dataset.zip -d dataset
    !pip install pytorch-fid --quiet
Runtime ~10-20 min on a T4.
"""

import os, shutil, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
DRIVE     = "/content/drive/MyDrive/pix2pix_runs"
OUT       = f"{DRIVE}/evaluation"          # all outputs land here (on Drive)
COND_CKPT = f"{DRIVE}/conditional_final.pt"
BASE_CKPT = f"{DRIVE}/baseline_final.pt"   # retrained baseline; skipped if absent
DATA_DIR  = "dataset"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

GREEN=(0,170,0); WATER=(0,90,255); RESI=(255,190,0); COMM=(230,30,30)
INDU=(150,60,200); OTHER=(240,240,240); ROAD=(140,140,140)
PALETTE={"green":GREEN,"water":WATER,"resi":RESI,"comm":COMM,
         "indu":INDU,"other":OTHER,"road":ROAD,"bg":(0,0,0)}
LVL=["low","med","high"]

# ---------------------------------------------------------------- model
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

def load_G(ckpt):
    G=UNetGenerator().to(DEVICE)
    ck=torch.load(ckpt,map_location=DEVICE)
    G.load_state_dict(ck["G"]); G.eval()
    return G

# ---------------------------------------------------------------- helpers
man=pd.read_csv(f"{DATA_DIR}/manifest.csv")
test=man[man.split=="test"].reset_index(drop=True)
print(f"test canvases: {len(test)}")

def load_input(tid):
    a=np.asarray(Image.open(f"{DATA_DIR}/test/input/tile_{tid}.png").convert("RGB"),
                 dtype=np.float32)
    return torch.from_numpy(a/127.5-1.0).permute(2,0,1)

def to_img(t):
    return ((t+1)*127.5).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)

def gen_batch(G,xs,walks,floods,zero_labels=False):
    """xs: [B,3,256,256]; walks/floods: int lists. Returns [B,3,256,256] cpu."""
    B=xs.shape[0]
    lab=torch.zeros(B,6)
    if not zero_labels:
        for i,(w,f) in enumerate(zip(walks,floods)):
            lab[i,w]=1.0; lab[i,3+f]=1.0
    maps=lab.view(B,6,1,1).expand(B,6,256,256)
    with torch.no_grad():
        return G(torch.cat([xs,maps],1).to(DEVICE)).cpu()

COLS=np.array(list(PALETTE.values())); NAMES=list(PALETTE.keys())
def measure(img):
    d=np.linalg.norm(img.reshape(-1,1,3).astype(int)-COLS.reshape(1,-1,3),axis=2)
    idx=d.argmin(1); f={n:(idx==i).mean() for i,n in enumerate(NAMES)}
    parts=np.array([f["resi"],f["comm"],f["indu"],f["other"]])
    p=parts/parts.sum() if parts.sum()>0 else parts
    ent=-(p[p>0]*np.log(p[p>0])).sum()/np.log(4) if parts.sum()>0 else 0.0
    return {"green":f["green"],"building":parts.sum(),"mix":ent}

def sweep(G,name):
    """All 9 combos on every test canvas. Returns tidy dataframe."""
    rows=[]
    BS=16
    for w in range(3):
        for fl in range(3):
            for s in range(0,len(test),BS):
                chunk=test.iloc[s:s+BS]
                xs=torch.stack([load_input(int(t)) for t in chunk.tile_id])
                out=gen_batch(G,xs,[w]*len(chunk),[fl]*len(chunk))
                for j,(_,r) in enumerate(chunk.iterrows()):
                    m=measure(to_img(out[j]))
                    m.update({"tile":int(r.tile_id),"walk":LVL[w],"flood":LVL[fl],
                              "model":name}); rows.append(m)
            print(f"  {name}: walk={LVL[w]} flood={LVL[fl]} done")
    return pd.DataFrame(rows)

# ================================================================ RUN
Gc=load_G(COND_CKPT); print(f"loaded conditional: {COND_CKPT}")
has_base=os.path.exists(BASE_CKPT)
if has_base:
    Gb=load_G(BASE_CKPT); print(f"loaded baseline:    {BASE_CKPT}")
else:
    print("no baseline_final.pt found — baseline sweep skipped (ablation still runs)")

# ---- Experiment 2+3: full sweep(s) --------------------------------------
print("\nExperiment 2/3: 9-combo sweep over all test canvases...")
df=sweep(Gc,"conditional")
if has_base: df=pd.concat([df,sweep(Gb,"baseline")])
df.to_csv(f"{OUT}/sweep_metrics.csv",index=False)

summary_lines=[]
for model in df.model.unique():
    d=df[df.model==model]
    wm=d.groupby("walk")["mix"].mean().reindex(LVL)
    gf=d.groupby("flood")["green"].mean().reindex(LVL)
    bw=d.groupby("walk")["building"].mean().reindex(LVL)
    bf=d.groupby("flood")["building"].mean().reindex(LVL)
    s=(f"\n=== {model} (n={len(d)//9} canvases x 9 combos) ===\n"
       f"walk->mix:     {wm.round(3).tolist()}  delta {wm['high']-wm['low']:+.3f}  "
       f"monotonic {wm.is_monotonic_increasing}\n"
       f"flood->green:  {gf.round(3).tolist()}  delta {gf['high']-gf['low']:+.3f}  "
       f"monotonic {gf.is_monotonic_increasing}\n"
       f"walk->building:{bw.round(3).tolist()}   flood->building:{bf.round(3).tolist()}")
    print(s); summary_lines.append(s)

# ---- tradeoff heatmaps (conditional) ------------------------------------
dc=df[df.model=="conditional"]
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for ax,metric,title in zip(axes,["green","mix","building"],
        ["Green %","Land-use mix entropy","Building coverage (emergent density)"]):
    M=dc.pivot_table(index="walk",columns="flood",values=metric,aggfunc="mean")
    M=M.reindex(index=LVL,columns=LVL)
    im=ax.imshow(M.values,cmap="viridis")
    ax.set_xticks(range(3),LVL); ax.set_yticks(range(3),LVL)
    ax.set_xlabel("flood label"); ax.set_ylabel("walkability label"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{M.values[i,j]:.3f}",ha="center",va="center",
                    color="white",fontsize=9)
    plt.colorbar(im,ax=ax,shrink=0.8)
plt.tight_layout(); plt.savefig(f"{OUT}/tradeoff_maps.png",dpi=130); plt.close()
print(f"saved {OUT}/tradeoff_maps.png")

# ---- Experiment 1+4: FID (+ ablation) -----------------------------------
print("\nExperiment 1/4: FID (real vs conditional / ablation"
      + (" / baseline" if has_base else "") + ")...")
dirs={"real":f"{OUT}/fid_real","conditional":f"{OUT}/fid_cond",
      "ablation":f"{OUT}/fid_abl"}
if has_base: dirs["baseline"]=f"{OUT}/fid_base"
for d in dirs.values():
    shutil.rmtree(d,ignore_errors=True); os.makedirs(d)

BS=16
for s in range(0,len(test),BS):
    chunk=test.iloc[s:s+BS]
    xs=torch.stack([load_input(int(t)) for t in chunk.tile_id])
    walks=[int(r.walk_lab) for _,r in chunk.iterrows()]
    floods=[int(r.flood_lab) for _,r in chunk.iterrows()]
    oc=gen_batch(Gc,xs,walks,floods)                       # true labels
    oa=gen_batch(Gc,xs,walks,floods,zero_labels=True)      # ablation
    ob=gen_batch(Gb,xs,walks,floods) if has_base else None
    for j,(_,r) in enumerate(chunk.iterrows()):
        tid=int(r.tile_id)
        shutil.copy(f"{DATA_DIR}/test/target/tile_{tid}.png",f"{dirs['real']}/{tid}.png")
        Image.fromarray(to_img(oc[j])).save(f"{dirs['conditional']}/{tid}.png")
        Image.fromarray(to_img(oa[j])).save(f"{dirs['ablation']}/{tid}.png")
        if has_base: Image.fromarray(to_img(ob[j])).save(f"{dirs['baseline']}/{tid}.png")

def fid(a,b):
    r=subprocess.run(["python","-m","pytorch_fid",a,b,"--device",DEVICE],
                     capture_output=True,text=True)
    for line in r.stdout.splitlines():
        if "FID" in line: return float(line.split()[-1])
    raise RuntimeError(r.stdout+r.stderr)

fids={}
for name in [k for k in dirs if k!="real"]:
    fids[name]=fid(dirs["real"],dirs[name])
    print(f"  FID real vs {name}: {fids[name]:.2f}")
s="\n=== FID (lower=more realistic) ===\n"+ \
  "\n".join(f"{k}: {v:.2f}" for k,v in fids.items())
summary_lines.append(s)

# ---- write summary ------------------------------------------------------
with open(f"{OUT}/RESULTS_SUMMARY.txt","w") as fh:
    fh.write("\n".join(summary_lines)+"\n")
print(f"\nAll outputs in {OUT}: sweep_metrics.csv, tradeoff_maps.png, "
      f"RESULTS_SUMMARY.txt, fid image folders")
