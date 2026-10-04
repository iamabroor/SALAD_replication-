#!/usr/bin/env python
"""
Paper-style qualitative figures for SALAD (cf. Fig. 5 of the paper).

For each selected test image:
  Image | Composition map C | Appearance map A_a | Composition anomaly map A_c | SALAD (A_a + A_c) | Ground truth

Usage (run from anywhere; uses your trained models):
  ~/salad/venv/bin/python ~/visualize_salad.py --category juice_bottle
  ~/salad/venv/bin/python ~/visualize_salad.py --category all

Options:
  --results  folder with <category>/teacher_final.pth etc. (default ~/salad/output/results)
  --n        images per type (good / logical / structural), default 3
  --select   random | best  (best = highest-scoring anomalies, like typical paper figures)
"""
import argparse, glob, os, random, sys
import numpy as np

REPO = os.path.expanduser("~/salad/SALAD")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from salad_dataset import (ImageFolderWithoutTarget, ImageFolderWithPath,
                           ImageFolderWithoutTargetWithSeg, ImageFolderWithPathWithSeg)
from train_salad import (teacher_normalization, map_normalization, score_normalization,
                         extract_features_mahalanobis, map_normalization_mahalanobis,
                         predict, predict_comp_map, predict_mahalanobis,
                         train_transform, default_transform)

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
TYPES = ["good", "logical_anomalies", "structural_anomalies"]
PALETTE = ListedColormap(["#000000", "#ccf1e3", "#708e12", "#fe0817", "#cf9554", "#ca18d6",
                          "#e6c025", "#f15044", "#4a7f00", "#0251d8"])
on_gpu = torch.cuda.is_available()


def build(category, results_dir, loco, maps):
    tr = ImageFolderWithoutTarget(os.path.join(loco, category, "train"), transform=transforms.Lambda(train_transform))
    trs = ImageFolderWithoutTarget(os.path.join(maps, category, "train"), transform=transforms.Lambda(train_transform))
    tr.seg, trs.seg = False, True
    te = ImageFolderWithPath(os.path.join(loco, category, "test"), transform=default_transform)
    tes = ImageFolderWithoutTarget(os.path.join(maps, category, "test"), transform=default_transform)
    te.seg, tes.seg = False, True
    train_set = ImageFolderWithoutTargetWithSeg(tr, trs)
    test_set = ImageFolderWithPathWithSeg(te, tes)
    va = ImageFolderWithoutTarget(os.path.join(loco, category, "validation"), transform=transforms.Lambda(train_transform))
    vas = ImageFolderWithoutTarget(os.path.join(maps, category, "validation"), transform=transforms.Lambda(train_transform))
    va.seg, vas.seg = False, True
    val_set = ImageFolderWithoutTargetWithSeg(va, vas)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1)
    d = os.path.join(results_dir, category)
    m = {k: torch.load(f"{d}/{f}", weights_only=False) for k, f in [
        ("teacher", "teacher_final.pth"), ("student", "student_final.pth"), ("autoencoder", "autoencoder_final.pth"),
        ("comp_ae", "comp_autoencoder_final.pth"), ("comp_unet", "comp_unet_final.pth")]}
    for v in m.values():
        if on_gpu: v.cuda()
        v.eval()
    t_mean, t_std = teacher_normalization(m["teacher"], train_loader)
    q_st_s, q_st_e, q_ae_s, q_ae_e = map_normalization(
        validation_loader=val_loader, teacher=m["teacher"], student=m["student"], autoencoder=m["autoencoder"],
        teacher_mean=t_mean, teacher_std=t_std, desc="Map normalization")
    q_eff_s, q_eff_e, q_seg_s, q_seg_e = score_normalization(
        validation_loader=val_loader, teacher=m["teacher"], student=m["student"], comp_ae=m["comp_ae"],
        comp_unet=m["comp_unet"], autoencoder=m["autoencoder"], teacher_mean=t_mean, teacher_std=t_std,
        q_st_start=q_st_s, q_st_end=q_st_e, q_ae_start=q_ae_s, q_ae_end=q_ae_e, desc="Score normalization")
    fv = extract_features_mahalanobis(train_loader, train_set, m["student"], t_mean, t_std)
    q_mah_s, q_mah_e = map_normalization_mahalanobis(val_loader, m["student"], t_mean, t_std, fv[1], fv[0], fv[3], fv[2], fv[5], fv[4])

    def run(idx):
        image, seg, _, path = test_set[idx]
        image = image.unsqueeze(0)
        if on_gpu: image, seg = image.cuda(), seg.cuda()
        a_a, _, _ = predict(image=image, teacher=m["teacher"], student=m["student"], autoencoder=m["autoencoder"],
                            teacher_mean=t_mean, teacher_std=t_std, q_st_start=q_st_s, q_st_end=q_st_e,
                            q_ae_start=q_ae_s, q_ae_end=q_ae_e)
        a_a = ((a_a - q_eff_s) / q_eff_e)[0, 0].cpu().numpy()
        a_c = predict_comp_map(seg, m["comp_ae"], m["comp_unet"]).unsqueeze(0)
        a_c = ((a_c - q_seg_s) / q_seg_e)[0, 0].cpu().numpy()
        mah = predict_mahalanobis(image=image, seg=seg, teacher=m["student"], teacher_mean=t_mean,
                                  feature_vectors_covinv=fv[1], feature_vectors_mean=fv[0],
                                  feature_vectors_covinv_seg=fv[3], feature_vectors_mean_seg=fv[2],
                                  feature_vectors_covinv_seg_area=fv[5], feature_vectors_mean_seg_area=fv[4],
                                  teacher_std=t_std, q_start=q_mah_s, q_end=q_mah_e)
        score = float(np.max(a_a) + mah + np.max(a_c))
        comp = seg.argmax(0).cpu().numpy()
        with torch.no_grad():
            c_rec = m["comp_ae"](seg.unsqueeze(0)).softmax(dim=1).argmax(dim=1)[0].cpu().numpy()
        return dict(path=path, a_a=a_a, a_c=a_c, comp=comp, c_rec=c_rec, score=score,
                    s_app=float(np.max(a_a)), s_comp=float(np.max(a_c)), s_glob=float(mah),
                    type=os.path.basename(os.path.dirname(path)))
    return test_set, run


def gt_mask(loco, category, path):
    typ, stem = os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]
    mask = np.zeros((256, 256), dtype=np.float32)
    for f in glob.glob(os.path.join(loco, category, "ground_truth", typ, stem, "*.png")):
        mask = np.maximum(mask, np.array(Image.open(f).convert("L").resize((256, 256), Image.NEAREST)) > 0)
    return mask


def pick(test_set, run, n, select, seed=0):
    by_type = {t: [] for t in TYPES}
    for i, (p, _) in enumerate(test_set.dataset1.samples):
        by_type[os.path.basename(os.path.dirname(p))].append(i)
    rng = random.Random(seed)
    chosen = []
    for t in TYPES:
        idx = by_type[t]
        if select == "best" and t != "good":
            res = sorted((run(i) for i in idx), key=lambda r: -r["score"])[:n]
        elif select == "best":
            res = sorted((run(i) for i in idx), key=lambda r: r["score"])[:n]
        else:
            res = [run(i) for i in rng.sample(idx, min(n, len(idx)))]
        chosen += res
    return chosen


def figure(category, samples, loco, out):
    rows = ["Image", "Composition map C", "Appearance A$_a$", "Composition A$_c$", "SALAD (A$_a$+A$_c$)", "Ground truth"]
    sal = [np.clip(s["a_a"], 0, None) + np.clip(s["a_c"], 0, None) for s in samples]
    vmax = max(np.percentile(np.stack(sal), 99.9), 1e-6)
    fig, ax = plt.subplots(len(rows), len(samples), figsize=(1.9 * len(samples), 1.9 * len(rows)))
    for j, (s, sm) in enumerate(zip(samples, sal)):
        img = np.array(Image.open(s["path"]).convert("RGB").resize((256, 256)))
        panels = [(img, None), (s["comp"], PALETTE), (np.clip(s["a_a"], 0, None), "jet"),
                  (np.clip(s["a_c"], 0, None), "jet"), (sm, "overlay"), (gt_mask(loco, category, s["path"]), "gray")]
        for i, (p, cm) in enumerate(panels):
            a = ax[i, j]; a.axis("off")
            if cm is None: a.imshow(p)
            elif cm == "overlay":
                a.imshow(img); a.imshow(p, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
            elif cm is PALETTE: a.imshow(p, cmap=PALETTE, vmin=0, vmax=9, interpolation="nearest")
            elif cm == "gray": a.imshow(p, cmap="gray", vmin=0, vmax=1)
            else: a.imshow(p, cmap=cm, vmin=0, vmax=vmax)
        label = {"good": "good", "logical_anomalies": "logical", "structural_anomalies": "structural"}[s["type"]]
        ax[0, j].set_title(f"{label}\nscore {s['score']:.2f}", fontsize=8)
    for i, r in enumerate(rows):
        ax[i, 0].text(-0.08, 0.5, r, transform=ax[i, 0].transAxes, rotation=90, ha="right", va="center", fontsize=8)
    fig.suptitle(f"SALAD - {category}", fontsize=11)
    plt.tight_layout()
    f = os.path.join(out, f"{category}_qualitative.png")
    plt.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="juice_bottle")
    ap.add_argument("--results", default=os.path.expanduser("~/salad/output/results"))
    ap.add_argument("--loco", default=f"{REPO}/data/mvtec_loco")
    ap.add_argument("--maps", default=f"{REPO}/data/mvtec_loco_composition_maps")
    ap.add_argument("--out", default=os.path.expanduser("~/salad/output/figures"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--select", choices=["random", "best"], default="random")
    ap.add_argument("--gallery", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--images", nargs="+", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    RES = {c: a.results for c in CATS}
    cats = CATS if a.category == "all" else [a.category]
    cats = [c for c in cats if os.path.exists(f"{RES[c]}/{c}/comp_unet_final.pth")]
    if not cats:
        sys.exit(f"No trained models found in {a.results}")
    all_samples = {}
    for c in cats:
        print(f"\n=== {c} ===")
        test_set, run = build(c, RES[c], a.loco, a.maps)
        with torch.no_grad():
            samples = pick(test_set, run, a.n, a.select)
        figure(c, samples, a.loco, a.out)
        all_samples[c] = samples

if __name__ == "__main__":
    main()
