"""Final SALAD results incl. extra seeds. Run: ~/salad/venv/bin/python ~/final_results.py"""
import re, os, glob, csv, sys
RERUN = "--rerun" in sys.argv   # use only the newest re-run for categories that were re-trained
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = os.path.expanduser("~/salad/output/logs")
OUT = os.path.expanduser("~/salad/output/final_report")
os.makedirs(OUT, exist_ok=True)
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PAPER = {"breakfast_box": (92.9, 85.7, 89.3), "juice_bottle": (99.7, 99.6, 99.7), "pushpins": (100.0, 98.8, 99.4),
         "screw_bag": (93.9, 96.0, 95.0), "splicing_connectors": (96.0, 98.6, 97.3)}


def final(f):
    t = open(f, errors="ignore").read()
    if "Final image auc" not in t:
        return None
    m = re.findall(r"All Logical: ([\d.]+), Struct: ([\d.]+)", t)
    l, s = map(float, m[-1])
    return l, s, (l + s) / 2


rows, A = [], []
SEEDVALS = {}
print(f"\n{'category':22s}{'seeds':>8s}{'logical':>10s}{'struct':>9s}{'mean':>8s}{'std':>7s}  | paper L / S / mean   | gap")
for c in CATS:
    runs = []
    for f in [f"{L}/{c}_train.log"] + sorted(glob.glob(f"{L}/{c}_seed*_train.log")):
        if os.path.exists(f):
            r = final(f)
            if r:
                sd = re.search(r"_seed(\d+)_", f)
                runs.append(("42" if not sd else sd.group(1),) + r)
    if not runs:
        print(f"{c:22s}  (no finished run)")
        continue
    if RERUN and len(runs) > 1:
        runs = [runs[-1]]
    lg = np.mean([x[1] for x in runs]); st = np.mean([x[2] for x in runs])
    mn = np.array([x[3] for x in runs]); sdv = mn.std(ddof=1) if len(mn) > 1 else 0.0
    p = PAPER[c]; seeds = ",".join(x[0] for x in runs)
    print(f"{c:22s}{seeds:>8s}{lg:10.2f}{st:9.2f}{mn.mean():8.2f}{sdv:7.2f}  | {p[0]:5.1f} / {p[1]:5.1f} / {p[2]:5.1f} | {mn.mean() - p[2]:+.2f}")
    if len(runs) > 1:
        for x in runs:
            print(f"{'':24s}seed {x[0]:>3s}: logical {x[1]:.2f}  struct {x[2]:.2f}  mean {x[3]:.2f}")
    SEEDVALS[c] = [x[3] for x in runs]
    rows.append([c, seeds, round(lg, 2), round(st, 2), round(mn.mean(), 2), round(sdv, 2), p[0], p[1], p[2], round(mn.mean() - p[2], 2)])
    A.append((lg, st, mn.mean()))

if len(A) == 5:
    a = np.mean(A, 0)
    print(f"{'AVERAGE (5 cats)':22s}{'':>8s}{a[0]:10.2f}{a[1]:9.2f}{a[2]:8.2f}{'':>7s}  |  96.5 /  95.7 /  96.1 | {a[2] - 96.1:+.2f}")
    rows.append(["AVERAGE", "", round(a[0], 2), round(a[1], 2), round(a[2], 2), "", 96.5, 95.7, 96.1, round(a[2] - 96.1, 2)])

SUF = "_rerun" if RERUN else "_with_seeds"
with open(f"{OUT}/final_results{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "seeds", "logical", "structural", "mean", "std_mean", "paper_logical", "paper_structural", "paper_mean", "gap_mean"])
    w.writerows(rows)

cr = [r for r in rows if r[0] != "AVERAGE"]
x = np.arange(len(cr))
fig, ax = plt.subplots(figsize=(10, 4.8))
b1 = ax.bar(x - 0.2, [r[4] for r in cr], 0.4, label="This replication" + (" (screw bag: re-run)" if RERUN else " (mean over seeds)"), color="#2a6f97")
b2 = ax.bar(x + 0.2, [PAPER[r[0]][2] for r in cr], 0.4, label="SALAD paper", color="#e07a5f")
ax.bar_label(b2, fmt="%.1f", fontsize=8)
for i, r in enumerate(cr):
    v = SEEDVALS[r[0]]
    if len(v) > 1:
        ax.scatter([x[i] - 0.2] * len(v), v, color="black", zorder=3, s=22, label="individual seeds" if i == 3 or not any(len(SEEDVALS[q[0]]) > 1 for q in cr[:i]) else None)
        for val in v:
            ax.annotate(f"{val:.1f}", (x[i] - 0.2, val), xytext=(-22, 0), textcoords="offset points", fontsize=7, va="center")
        ax.annotate(f"mean {r[4]:.1f}", (x[i] - 0.2, max(v)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    else:
        ax.annotate(f"{r[4]:.1f}", (x[i] - 0.2, r[4]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(x, [r[0].replace("_", " ") for r in cr]); ax.set_ylim(80, 102); ax.set_ylabel("Image AUROC (%)")
ax.set_title(f"MVTec LOCO - avg: ours {rows[-1][4]:.2f} vs paper 96.1" if rows[-1][0] == "AVERAGE" else "MVTec LOCO")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(f"{OUT}/final_results{SUF}.png", dpi=180)
print(f"\nSaved {OUT}/final_results{SUF}.csv and .png")
