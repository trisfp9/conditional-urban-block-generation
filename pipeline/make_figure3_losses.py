"""
make_figure3.py  -- Figure 3, training dynamics of both models.
Run in Colab after mounting Drive. Produces a single-column (3.4 in) figure
at 600 dpi, suitable for the template.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVE = "/content/drive/MyDrive/pix2pix_runs"
cond = pd.read_csv(f"{DRIVE}/conditional_losses.csv")
base = pd.read_csv(f"{DRIVE}/baseline_losses.csv")

# SAFETY CHECK: make sure this is the RETRAINED baseline, not the collapsed run.
alive = (base.D > 0.01).sum()
print(f"baseline discriminator active in {alive}/{len(base)} epochs")
assert alive > 100, ("This looks like the COLLAPSED baseline run. Use the "
                     "retrained file (D active in ~192/200 epochs).")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True)

ax1.plot(cond.epoch, cond.G, linewidth=0.9, label="Conditional")
ax1.plot(base.epoch, base.G, linewidth=0.9, linestyle="--", label="Unconditional")
ax1.set_ylabel("Generator loss", fontsize=8)
ax1.legend(fontsize=7, frameon=False)
ax1.tick_params(labelsize=7)

ax2.plot(cond.epoch, cond.D, linewidth=0.9, label="Conditional")
ax2.plot(base.epoch, base.D, linewidth=0.9, linestyle="--", label="Unconditional")
ax2.set_ylabel("Discriminator loss", fontsize=8)
ax2.set_xlabel("Epoch", fontsize=8)
ax2.tick_params(labelsize=7)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{DRIVE}/figure3_losses.png", dpi=600, bbox_inches="tight")
plt.savefig("figure3_losses.png", dpi=600, bbox_inches="tight")
print("saved figure3_losses.png to Drive and to the session")
