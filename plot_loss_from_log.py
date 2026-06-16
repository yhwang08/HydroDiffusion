#!/usr/bin/env python
"""
Overlay TRAINING (solid) and VALIDATION (dashed) loss curves for every
`global_diffusion_unet_static_200_generic_diffusion_unet_advanced*.out`
file on one axis, with identical colours per run.

Output:  all_loss_curves.png  in LOG_DIR
"""

import os
import re
import glob
import matplotlib.pyplot as plt
from itertools import cycle

# ------------------------------------------------------------------ #
LOG_DIR  = "/home/yihan/diffusion_ssm/logs/"
PATTERN  = "global_diffusion_*.out"
PREFIX   = "global_diffusion_"  # strip from legend labels
# ------------------------------------------------------------------ #

# 1) Collect log files
log_files = sorted(glob.glob(os.path.join(LOG_DIR, PATTERN)))
if not log_files:
    raise FileNotFoundError(f"No files matching '{PATTERN}' in {LOG_DIR}")

# 2) Regex patterns
pat_train = re.compile(r"Epoch\s+(\d+)\s+TRAINING loss:\s+([0-9.]+)")
pat_val   = re.compile(r"Epoch\s+(\d+)\s+VALIDATION loss:\s+([0-9.]+)")

def trim(name):
    name = os.path.basename(name).replace(".out", "")
    return name[len(PREFIX):] if name.startswith(PREFIX) else name

# 3) Parse logs → run → (epochs, train, val)
loss = {}
for path in log_files:
    ep, tr, va = [], [], []
    with open(path) as f:
        for line in f:
            mt = pat_train.search(line)
            mv = pat_val.search(line)
            if mt:
                ep.append(int(mt.group(1))); tr.append(float(mt.group(2)))
            if mv:
                va.append(float(mv.group(2)))
    if not ep or not va:
        print(f"[WARN] {path}: missing data, skipped.")
        continue
    n = min(len(tr), len(va))
    loss[trim(path)] = (ep[:n], tr[:n], va[:n])

# 4) Plot all on one axis, reuse colour between solid/dashed
fig, ax = plt.subplots(figsize=(15, 6))
ax.set(title="Training vs Validation losses (all runs)",
       xlabel="Epoch", ylabel="Loss")
#ax.set_xlim(0, 60)
ax.grid(alpha=0.3, linestyle="--")

palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

for run, (ep, tr, va) in loss.items():
    colour = next(palette)
    ax.plot(ep, tr, color=colour, linewidth=2, label=f"{run} – train")
    ax.plot(ep, va, color=colour, linestyle="--", linewidth=2,
            label=f"{run} – val")

ax.legend(fontsize=8, ncol=2)
fig.tight_layout()

out_png = os.path.join(LOG_DIR, "all_loss_curves.png")
fig.savefig(out_png, dpi=160)
print(f"Saved plot → {out_png}")
