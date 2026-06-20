"""Generate the figures for the EB-JEPA diagnosis PDF. Run: python make_jepa_figures.py
All numbers are the honest, verified values from JEPA_DIAGNOSIS.md (per-recording = 16-window
mean-pool; 3-seed where a std is shown). Output: figs/*.pdf (vector, for LaTeX inclusion)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight"})

STEEL, GREY, TOMATO, PURPLE, DARKRED, GREEN, ORANGE = (
    "#4C72B0", "#B0B0B0", "#E1574C", "#8172B3", "#8B0000", "#2E8B57", "#DD8452")


# ---- Fig 1: the plateau (our per-recording attempts vs references) ----
rows = [  # (label, bacc, color, hatch)
    ("VICReg + spectral 0.1  (soft)", 0.836, STEEL, "//"),
    ("SIGReg + corruption", 0.825, STEEL, None),
    ("VICReg + corruption  (3-seed)", 0.819, STEEL, None),
    ("VICReg + corruption → fine-tune", 0.812, "#7BA3D0", None),
    ("VICReg + corruption + multi-corpus 13k", 0.812, GREY, None),
    ("VICReg + corruption + bigger conv", 0.805, GREY, None),
    ("Transformer 3.65M + multi-corpus", 0.798, PURPLE, None),
    ("VICReg base (no corruption)", 0.796, GREY, None),
    ("VICReg + corruption, inv_coeff=25", 0.789, TOMATO, None),
    ("Masked-JEPA (EMA predictor)", 0.680, DARKRED, None),
]
labels = [r[0] for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
hatch = [r[3] for r in rows][::-1]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.axvspan(0.795, 0.84, color="#F2E8C9", alpha=0.5, zorder=0, label="0.4M-conv plateau (0.80–0.84)")
y = np.arange(len(labels))
bars = ax.barh(y, vals, color=cols, hatch=hatch, edgecolor="white", zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(0.846, ls="--", color=GREEN, lw=1.8, zorder=4)
ax.text(0.846, len(labels) - 0.3, " LaBraM-Base 0.846", color=GREEN, fontsize=9, va="top")
ax.axvline(0.812, ls=":", color=ORANGE, lw=1.8, zorder=4)
ax.text(0.812, -0.7, "EEGNet (supervised) 0.812", color=ORANGE, fontsize=9, ha="center")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0.66, 0.875); ax.set_xlabel("Per-recording balanced accuracy")
ax.set_title("Every EB-JEPA variant plateaus on a 0.4M conv — and only a foundation model clears it",
             fontsize=10.5, weight="bold")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
fig.savefig(os.path.join(OUT, "fig_plateau.pdf")); plt.close(fig)


# ---- Fig 2: what moved the needle (deltas in pp on per-rec BAcc, from 0.4M-conv baseline) ----
levers = [("Corruption augmentation", +2.3, GREEN),
          ("inv_coeff 25 → 1", +3.0, GREEN),
          ("Fine-tune vs frozen", -0.7, GREY),
          ("4× data (multi-corpus)", -0.7, GREY),
          ("Scale conv ↑ (bigger)", -1.4, TOMATO)]
labels2 = [l[0] for l in levers][::-1]
deltas = [l[1] for l in levers][::-1]
cols2 = [l[2] for l in levers][::-1]
fig, ax = plt.subplots(figsize=(7.6, 3.2))
y = np.arange(len(labels2))
ax.barh(y, deltas, color=cols2, edgecolor="white", zorder=3)
ax.axvline(0, color="k", lw=0.8)
for yi, d in zip(y, deltas):
    ax.text(d + (0.12 if d >= 0 else -0.12), yi, f"{d:+.1f} pp",
            va="center", ha="left" if d >= 0 else "right", fontsize=9.5)
ax.set_yticks(y); ax.set_yticklabels(labels2, fontsize=9.5)
ax.set_xlim(-2.4, 4.2); ax.set_xlabel("Δ per-recording balanced accuracy (percentage points)")
ax.set_title("What moved the needle — and what didn't", fontsize=10.5, weight="bold")
fig.savefig(os.path.join(OUT, "fig_levers.pdf")); plt.close(fig)


# ---- Fig 3: per-window — our JEPA in the literature field ----
field = [("LaBraM-Base", 0.814, GREY), ("FEMBA-Base", 0.811, GREY), ("AFTA", 0.800, GREY),
         ("EEGPT", 0.798, GREY), ("ST-Transformer", 0.797, GREY), ("BIOT", 0.796, GREY),
         ("SPaRCNet", 0.790, GREY), ("FFCL", 0.785, GREY), ("CNN-Transformer", 0.778, GREY),
         ("ContraWR", 0.775, GREY), ("EB-JEPA (ours, SIGReg+corrupt)", 0.775, STEEL)]
field.sort(key=lambda r: r[1])
labels3 = [r[0] for r in field]; vals3 = [r[1] for r in field]; cols3 = [r[2] for r in field]
fig, ax = plt.subplots(figsize=(7.8, 4.0))
y = np.arange(len(labels3))
ax.barh(y, vals3, color=cols3, edgecolor="white", zorder=3)
for yi, v, lab in zip(y, vals3, labels3):
    ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=8.5,
            weight="bold" if "ours" in lab else "normal")
ax.set_yticks(y); ax.set_yticklabels(labels3, fontsize=9)
ax.get_yticklabels()[[i for i, l in enumerate(labels3) if "ours" in l][0]].set_fontweight("bold")
ax.set_xlim(0.76, 0.822); ax.set_xlabel("Per-window balanced accuracy (literature convention)")
ax.set_title("Per-window: our SSL sits in the lower-middle — ~4 pp below LaBraM",
             fontsize=10.5, weight="bold")
fig.savefig(os.path.join(OUT, "fig_perwindow.pdf")); plt.close(fig)


# ---- Fig 4: the inv_coeff finding (3-seed, error bars) ----
groups = ["Per-recording BAcc", "Per-window BAcc"]
inv1 = [0.819, 0.770]; inv1e = [0.004, 0.0]
inv25 = [0.789, 0.734]; inv25e = [0.005, 0.0]
x = np.arange(len(groups)); w = 0.34
fig, ax = plt.subplots(figsize=(6.2, 3.6))
b1 = ax.bar(x - w/2, inv1, w, yerr=inv1e, capsize=4, color=STEEL, label="inv_coeff = 1  (ours)")
b2 = ax.bar(x + w/2, inv25, w, yerr=inv25e, capsize=4, color=TOMATO,
            label="inv_coeff = 25  (VICReg paper default)")
for b in (b1, b2):
    for r in b:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.006,
                f"{r.get_height():.3f}", ha="center", fontsize=9)
for xi, a, bb in zip(x, inv1, inv25):
    ax.annotate(f"−{(a-bb)*100:.1f} pp", xy=(xi, max(a, bb) + 0.03), ha="center",
                fontsize=9.5, weight="bold", color=DARKRED)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0.70, 0.90); ax.set_ylabel("Balanced accuracy")
ax.set_title("Surprise: VICReg's paper-default invariance weight is WORSE on EEG",
             fontsize=10.5, weight="bold")
ax.legend(fontsize=9, loc="upper right")
fig.savefig(os.path.join(OUT, "fig_invcoeff.pdf")); plt.close(fig)

print("wrote 4 figures to", OUT)
