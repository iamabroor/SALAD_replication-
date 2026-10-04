#!/usr/bin/env python
"""
Builds the SALAD replication report from the training logs.
Output folder (default ~/salad/output/final_report):
  results_table.csv / .md        - per-category logical / structural / mean vs paper (+ average)
  branches_table.csv / .md       - single-branch AUROC vs paper appendix Table 2
  training_table.csv             - AUROC at every 10k-step checkpoint
  charts/*.png                   - comparison bar charts, branch chart, training curves, loss curves
  report.html                    - everything on one page (open in a browser)
"""
import argparse, csv, datetime, glob, html, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SHORT = {"breakfast_box": "Breakfast box", "juice_bottle": "Juice bottle", "pushpins": "Pushpins",
         "screw_bag": "Screw bag", "splicing_connectors": "Splicing conn."}
PAPER = {"breakfast_box": (92.9, 85.7, 89.3), "juice_bottle": (99.7, 99.6, 99.7), "pushpins": (100.0, 98.8, 99.4),
         "screw_bag": (93.9, 96.0, 95.0), "splicing_connectors": (96.0, 98.6, 97.3)}
PAPER_AVG = (96.5, 95.7, 96.1)
PAPER_BR = {"appearance": {"breakfast_box": 85.7, "juice_bottle": 96.9, "pushpins": 96.8, "screw_bag": 77.9, "splicing_connectors": 96.6},
            "composition": {"breakfast_box": 77.1, "juice_bottle": 87.0, "pushpins": 87.7, "screw_bag": 88.2, "splicing_connectors": 86.2},
            "global": {"breakfast_box": 82.2, "juice_bottle": 97.7, "pushpins": 91.8, "screw_bag": 86.3, "splicing_connectors": 87.3}}
OTHERS = {"Reproduction A (issue #5)": {"breakfast_box": 84.74, "juice_bottle": 98.46, "pushpins": 98.16, "screw_bag": 93.13, "splicing_connectors": 96.52},
          "Reproduction B (issue #9)": {"breakfast_box": 85.81, "juice_bottle": 99.61, "pushpins": 90.23, "screw_bag": 87.43, "splicing_connectors": 96.22},
          "Released weights (issue #6)": {"juice_bottle": 99.90, "pushpins": 95.60, "screw_bag": 83.71, "splicing_connectors": 96.76}}

C_OURS, C_PAPER, C_GREY = "#2a6f97", "#e07a5f", "#9aa5b1"


def parse_log(path):
    t = open(path, errors="ignore").read()
    evals = re.findall(r"All Logical: ([\d.]+), Struct: ([\d.]+)\s*\n"
                       r"Maha Logical: ([\d.]+), Struct: ([\d.]+)\s*\n"
                       r"Img Logical: ([\d.]+), Struct: ([\d.]+)\s*\n"
                       r"Comp Logical: ([\d.]+), Struct: ([\d.]+)", t)
    heads = re.findall(r"(Intermediate|Final) image auc: ([\d.]+)", t)
    rows = []
    for i, e in enumerate(evals):
        v = list(map(float, e))
        kind = heads[i][0] if i < len(heads) else "?"
        rows.append(dict(kind=kind, step=None, all_l=v[0], all_s=v[1], maha_l=v[2], maha_s=v[3],
                         img_l=v[4], img_s=v[5], comp_l=v[6], comp_s=v[7]))
    k = 0
    for r in rows:
        if r["kind"] == "Intermediate":
            k += 1; r["step"] = 10000 * k
        else:
            r["step"] = 70000
    flat = t.replace("\r", "\n")
    loss = [(int(s), float(a), float(b), float(c)) for a, b, c, s in re.findall(
        r"Current loss: ([\d.]+), comp recon loss ([\d.]+), comp disc loss ([\d.]+):[^\n]*?\|\s*(\d+)/70000", flat)]
    tq = re.findall(r"70000/70000 \[([\d:]+)<[^,]*,\s*([\d.]+)it/s", flat)
    final = next((r for r in reversed(rows) if r["kind"] == "Final"), None)
    return dict(rows=rows, final=final, loss=loss, time=tq[-1][0] if tq else None,
                speed=float(tq[-1][1]) if tq else None)


def mean(a, b): return (a + b) / 2


def write_table(path_csv, header, rows):
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)
    open(path_csv.replace(".csv", ".md"), "w").write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/salad/output/logs"))
    ap.add_argument("--out", default=os.path.expanduser("~/salad/output/final_report"))
    ap.add_argument("--record", default=os.path.expanduser("~/salad/output/run_record.txt"))
    ap.add_argument("--use", nargs="*", default=[], help="per-category log, e.g. screw_bag=seed1")
    a = ap.parse_args()
    USE = dict(u.split("=", 1) for u in a.use)
    os.makedirs(f"{a.out}/charts", exist_ok=True)

    data = {}
    for c in CATS:
        f = f"{a.logs}/{c}_{USE[c]}_train.log" if c in USE else f"{a.logs}/{c}_train.log"
        if os.path.exists(f):
            d = parse_log(f)
            if d["final"]: data[c] = d
    if not data:
        raise SystemExit("No finished categories found in " + a.logs)
    cats = [c for c in CATS if c in data]

    rows, L, S, M, PL, PS, PM = [], [], [], [], [], [], []
    for c in cats:
        r = data[c]["final"]; p = PAPER[c]; m = mean(r["all_l"], r["all_s"])
        rows.append([c + (" (re-run)" if c in USE else ""), f"{r['all_l']:.2f}", f"{r['all_s']:.2f}", f"{m:.2f}", p[0], p[1], p[2], f"{m - p[2]:+.2f}",
                     data[c]["time"] or "", f"{data[c]['speed']:.2f}" if data[c]["speed"] else ""])
        L.append(r["all_l"]); S.append(r["all_s"]); M.append(m); PL.append(p[0]); PS.append(p[1]); PM.append(p[2])
    avg = [np.mean(L), np.mean(S), np.mean(M)]
    pav = PAPER_AVG if len(cats) == 5 else (np.mean(PL), np.mean(PS), np.mean(PM))
    rows.append([f"AVERAGE ({len(cats)} categories)", f"{avg[0]:.2f}", f"{avg[1]:.2f}", f"{avg[2]:.2f}",
                 f"{pav[0]:.1f}", f"{pav[1]:.1f}", f"{pav[2]:.1f}", f"{avg[2] - pav[2]:+.2f}", "", ""])
    res_md = write_table(f"{a.out}/results_table.csv",
                         ["category", "logical", "structural", "mean", "paper_logical", "paper_structural",
                          "paper_mean", "gap_mean", "train_time_70k", "it_per_s"], rows)

    brow = []
    for c in cats:
        r = data[c]["final"]
        app, glo, com = mean(r["img_l"], r["img_s"]), mean(r["maha_l"], r["maha_s"]), mean(r["comp_l"], r["comp_s"])
        brow.append([c + (" (re-run)" if c in USE else ""), f"{app:.2f}", PAPER_BR["appearance"][c], f"{app - PAPER_BR['appearance'][c]:+.2f}",
                     f"{glo:.2f}", PAPER_BR["global"][c], f"{glo - PAPER_BR['global'][c]:+.2f}",
                     f"{com:.2f}", PAPER_BR["composition"][c], f"{com - PAPER_BR['composition'][c]:+.2f}"])
    br_md = write_table(f"{a.out}/branches_table.csv",
                        ["category", "appearance", "paper", "gap", "global", "paper", "gap", "composition", "paper", "gap"], brow)

    trows = []
    for c in cats:
        for r in data[c]["rows"]:
            trows.append([c, r["step"], r["kind"], f"{r['all_l']:.2f}", f"{r['all_s']:.2f}", f"{mean(r['all_l'], r['all_s']):.2f}",
                          f"{mean(r['img_l'], r['img_s']):.2f}", f"{mean(r['maha_l'], r['maha_s']):.2f}", f"{mean(r['comp_l'], r['comp_s']):.2f}"])
    write_table(f"{a.out}/training_table.csv",
                ["category", "step", "type", "logical", "structural", "mean", "appearance", "global", "composition"], trows)

    charts = []
    x = np.arange(len(cats)); lab = [SHORT[c] + ("\n(re-run)" if c in USE else "") for c in cats]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x - 0.2, M, 0.4, label="This replication", color=C_OURS)
    b2 = ax.bar(x + 0.2, PM, 0.4, label="SALAD paper", color=C_PAPER)
    ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, lab); ax.set_ylim(min(M + PM) - 8, 101.5); ax.set_ylabel("Image AUROC (%)")
    ax.set_title(f"MVTec LOCO - mean of logical & structural AUROC  (avg: ours {avg[2]:.1f} vs paper {pav[2]:.1f})")
    ax.legend(loc="lower left", frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); p = f"{a.out}/charts/01_mean_vs_paper.png"; plt.savefig(p, dpi=180); plt.close(); charts.append((p, "Per-category AUROC: this replication vs the paper"))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for axx, ours, pap, name in [(axs[0], L, PL, "Logical anomalies"), (axs[1], S, PS, "Structural anomalies")]:
        u = axx.bar(x - 0.2, ours, 0.4, label="This replication", color=C_OURS)
        v = axx.bar(x + 0.2, pap, 0.4, label="SALAD paper", color=C_PAPER)
        axx.bar_label(u, fmt="%.1f", fontsize=7); axx.bar_label(v, fmt="%.1f", fontsize=7)
        axx.set_xticks(x, lab, fontsize=8); axx.set_title(name); axx.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylim(min(L + S + PL + PS) - 8, 101.5); axs[0].set_ylabel("Image AUROC (%)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0.07, 1, 1)); p = f"{a.out}/charts/02_logical_structural.png"; plt.savefig(p, dpi=180); plt.close(); charts.append((p, "Logical and structural AUROC separately"))

    figs = []
    fdir = os.path.dirname(a.out.rstrip("/")) + "/figures"
    for sub, title in [("", "Best-scoring examples"), ("random/", "Random examples"), ("all/", "Overview")]:
        for f in sorted(glob.glob(f"{fdir}/{sub}*.png")):
            figs.append((os.path.relpath(f, a.out), f"{title}: {os.path.basename(f)}"))
    record = open(a.record).read() if os.path.exists(a.record) else "(run collect_run_info.sh to add the hardware/software record)"

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>SALAD replication report</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1d2733}}
h1{{margin-bottom:4px}} h2{{margin-top:36px;border-bottom:2px solid #e3e8ee;padding-bottom:4px}}
table{{border-collapse:collapse;margin:10px 0;font-size:14px}} th,td{{border:1px solid #d5dbe1;padding:5px 9px;text-align:right}}
th{{background:#f1f4f7}} td:first-child,th:first-child{{text-align:left}} img{{max-width:100%;border:1px solid #e3e8ee;margin:6px 0 18px}}
pre{{background:#f6f8fa;padding:12px;overflow-x:auto;font-size:12px}} .note{{color:#56636f;font-size:14px}}</style></head><body>
<h1>SALAD (ICCV 2025) - MVTec LOCO replication</h1>
<p class="note">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}.</p>
<h2>1. Results vs paper</h2>{res_md}
<h2>2. Charts</h2>{"".join(f"<h3>{html.escape(t)}</h3><img src='{os.path.relpath(p, a.out)}'>" for p, t in charts)}
<h2>3. Qualitative results</h2>{"".join(f"<h3>{html.escape(t)}</h3><img src='{s}'>" for s, t in figs) or "<p>(no figures found)</p>"}
<h2>4. Compute and software record</h2><pre>{html.escape(record)}</pre>
</body></html>"""
    open(f"{a.out}/report.html", "w").write(page)
    print(res_md); print(br_md)
    print(f"Report written to {a.out}/report.html  ({len(charts)} charts, {len(figs)} figures)")


if __name__ == "__main__":
    main()
