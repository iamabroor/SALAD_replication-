#!/usr/bin/env python
"""
pipeline_walkthrough.py  -  SALAD full pipeline per-image figures
==================================================================
Panels per image (15 total):
  1  RGB input
  2  SAM foreground mask          (background removed)
  3  DINO patch clusters          (K=6 KMeans on ViT-B/8 features)
  4  Pseudo-labels                (majority-vote per SAM-HQ mask)
  5  Composition map C            (UNet output)
  6  Normal C_rec                 (composition AE reconstruction)
  7  Teacher features             (EfficientNet PCA->RGB)
  8  Student features             (mimics teacher)
  9  Teacher-Student diff         (raw A_a signal)
 10  AE reconstruction            (normal appearance)
 11  Appearance map A_a           (teacher-student error)
 12  Composition anomaly A_c      (discriminator C vs C_rec)
 13  Mahalanobis heatmap          (global statistical branch)
 14  Fused overlay                (anomaly map on RGB)
 15  Ground truth mask

Usage:
  ~/salad/venv/bin/python ~/pipeline_walkthrough.py \
      --category juice_bottle \
      --select mixed \
      --loco ~/salad/SALAD/data/mvtec_loco \
      --maps ~/salad/SALAD/data/mvtec_loco_composition_maps \
      --results ~/salad/output/results \
      --out ~/salad/output/walkthrough
"""
import argparse, os, sys, glob, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from PIL import Image

REPO = os.path.expanduser("~/salad/SALAD")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from salad_dataset import (ImageFolderWithoutTarget, ImageFolderWithPath,
                            ImageFolderWithoutTargetWithSeg, ImageFolderWithPathWithSeg)
from train_salad import (teacher_normalization, map_normalization, score_normalization,
                         extract_features_mahalanobis, map_normalization_mahalanobis,
                         predict, predict_comp_map, predict_mahalanobis,
                         train_transform, default_transform)

ON_GPU = torch.cuda.is_available()
CATS   = ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
TYPES  = ["good","logical_anomalies","structural_anomalies"]

COMP_PAL = ListedColormap(["#13293d","#f0635f","#72c2b6","#e8b030","#f4d03f",
                            "#8e6bbf","#5dade2","#e67e22","#a3e4d7","#ffffff"])
CLUSTER_COLORS = np.array([
    [31,119,180],[255,127,14],[44,160,44],
    [214,39,40],[148,103,189],[140,86,75]], dtype=np.uint8)

PART_NAMES = {0:"background",1:"part-1",2:"part-2",3:"part-3",
              4:"part-4",5:"part-5",6:"part-6"}


# -- SAM + DINO setup ---------------------------------------------------------

_sam_model   = None
_dino_model  = None

def get_sam(device):
    global _sam_model
    if _sam_model is None:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        ckpt = os.path.expanduser("~/salad/SALAD/sam_vit_b_01ec64.pth")
        if not os.path.exists(ckpt):
            hits = glob.glob(os.path.join(REPO, "sam*.pth"))
            if not hits:
                print("WARNING: SAM checkpoint not found -- foreground mask will be all-ones")
                return None
            ckpt = hits[0]
        model_type = "vit_b" if "vit_b" in ckpt else ("vit_h" if "vit_h" in ckpt else "vit_b")
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device)
        sam.eval()
        _sam_model = sam
        print(f"SAM loaded: {ckpt}")
    return _sam_model


def get_dino(device):
    global _dino_model
    if _dino_model is None:
        import timm
        m = timm.create_model("vit_base_patch8_224_dino", pretrained=True)
        m.to(device)
        m.eval()
        _dino_model = m
        print("DINO ViT-B/8 loaded")
    return _dino_model


def sam_foreground(img_pil, device):
    """Returns a (H,W) bool mask where True = foreground."""
    sam = get_sam(device)
    if sam is None:
        return np.ones((img_pil.height, img_pil.width), dtype=bool)
    from segment_anything import SamAutomaticMaskGenerator
    img_np = np.array(img_pil.convert("RGB"))
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                    pred_iou_thresh=0.88, stability_score_thresh=0.95)
    masks = gen.generate(img_np)
    if not masks:
        return np.ones(img_np.shape[:2], dtype=bool)
    H, W = img_np.shape[:2]
    corner_mask = np.zeros((H, W), dtype=bool)
    for cy, cx in [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]:
        for m in masks:
            if m["segmentation"][cy, cx]:
                corner_mask |= m["segmentation"]
    fg = ~corner_mask
    if fg.sum() < 0.05 * H * W:
        fg = np.ones((H, W), dtype=bool)
    return fg


def dino_clusters(img_pil, device, k=6):
    """Returns (H, W) int array of cluster IDs 0..k-1 and (H,W,3) RGB colour image."""
    from sklearn.cluster import KMeans
    dino = get_dino(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = tf(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino.forward_features(x)          # (1, 1+28*28, C)
        patch_tokens = feats[:, 1:, :]            # drop CLS
    patch_tokens = patch_tokens[0].cpu().numpy()  # (784, C)
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = km.fit_predict(patch_tokens)         # (784,)
    grid = labels.reshape(28, 28).astype(np.uint8)
    from PIL import Image as PILImage
    grid_img = PILImage.fromarray(grid).resize((256, 256), PILImage.NEAREST)
    grid256  = np.array(grid_img)
    colour   = CLUSTER_COLORS[grid256 % len(CLUSTER_COLORS)]
    return grid256, colour


# -- model loading & normalisation --------------------------------------------

def load_models(results_dir, category):
    d = os.path.join(results_dir, category)
    names = [("teacher","teacher_final.pth"),("student","student_final.pth"),
             ("autoencoder","autoencoder_final.pth"),
             ("comp_ae","comp_autoencoder_final.pth"),
             ("comp_unet","comp_unet_final.pth")]
    m = {}
    for k, f in names:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            sys.exit(f"Model not found: {p}")
        m[k] = torch.load(p, weights_only=False)
        if ON_GPU: m[k] = m[k].cuda()
        m[k].eval()
    return m


def build_loaders(loco, maps, category):
    tr  = ImageFolderWithoutTarget(os.path.join(loco, category, "train"),
                                   transform=transforms.Lambda(train_transform))
    trs = ImageFolderWithoutTarget(os.path.join(maps, category, "train"),
                                   transform=transforms.Lambda(train_transform))
    tr.seg = False; trs.seg = True
    train_set = ImageFolderWithoutTargetWithSeg(tr, trs)

    te  = ImageFolderWithPath(os.path.join(loco, category, "test"),
                              transform=default_transform)
    tes = ImageFolderWithoutTarget(os.path.join(maps, category, "test"),
                                   transform=default_transform)
    te.seg = False; tes.seg = True
    test_set = ImageFolderWithPathWithSeg(te, tes)

    va  = ImageFolderWithoutTarget(os.path.join(loco, category, "validation"),
                                   transform=transforms.Lambda(train_transform))
    vas = ImageFolderWithoutTarget(os.path.join(maps, category, "validation"),
                                   transform=transforms.Lambda(train_transform))
    va.seg = False; vas.seg = True
    val_set = ImageFolderWithoutTargetWithSeg(va, vas)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1)
    return train_set, val_loader, train_loader, test_set


def normalise(results_dir, category, models, train_loader, val_loader):
    m = models
    t_mean, t_std = teacher_normalization(m["teacher"], train_loader)
    q_st_s,q_st_e,q_ae_s,q_ae_e = map_normalization(
        validation_loader=val_loader, teacher=m["teacher"], student=m["student"],
        autoencoder=m["autoencoder"], teacher_mean=t_mean, teacher_std=t_std,
        desc="Map norm")
    q_eff_s,q_eff_e,q_seg_s,q_seg_e = score_normalization(
        validation_loader=val_loader, teacher=m["teacher"], student=m["student"],
        comp_ae=m["comp_ae"], comp_unet=m["comp_unet"], autoencoder=m["autoencoder"],
        teacher_mean=t_mean, teacher_std=t_std,
        q_st_start=q_st_s, q_st_end=q_st_e, q_ae_start=q_ae_s, q_ae_end=q_ae_e,
        desc="Score norm")
    fv = extract_features_mahalanobis(train_loader, train_loader.dataset,
                                      m["student"], t_mean, t_std)
    q_mah_s,q_mah_e = map_normalization_mahalanobis(
        val_loader, m["student"], t_mean, t_std,
        fv[1],fv[0],fv[3],fv[2],fv[5],fv[4])
    return dict(t_mean=t_mean, t_std=t_std,
                q_st_s=q_st_s, q_st_e=q_st_e, q_ae_s=q_ae_s, q_ae_e=q_ae_e,
                q_eff_s=q_eff_s, q_eff_e=q_eff_e, q_seg_s=q_seg_s, q_seg_e=q_seg_e,
                fv=fv, q_mah_s=q_mah_s, q_mah_e=q_mah_e)


def feats_to_rgb(t):
    """Project (C,H,W) feature tensor to (H,W,3) uint8 via PCA."""
    from sklearn.decomposition import PCA
    arr = t.cpu().float().numpy()           # (C,H,W)
    C,H,W = arr.shape
    flat  = arr.reshape(C, H*W).T          # (H*W, C)
    n_comp = min(3, C, H*W)
    pca   = PCA(n_components=n_comp)
    proj  = pca.fit_transform(flat)        # (H*W, 3)
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0,0),(0,3-proj.shape[1])))
    proj  = proj.reshape(H, W, 3)
    lo, hi = proj.min(), proj.max()
    if hi > lo:
        proj = (proj - lo) / (hi - lo)
    rgb = (proj * 255).clip(0,255).astype(np.uint8)
    from PIL import Image as PILImage
    rgb = np.array(PILImage.fromarray(rgb).resize((256,256), PILImage.BILINEAR))
    return rgb


def mahal_heatmap(image, seg, models, norms, device):
    """Per-pixel Mahalanobis heatmap."""
    m, n = models, norms
    fv = n["fv"]
    comp = seg.argmax(0).cpu().numpy()          # (256,256) part IDs
    with torch.no_grad():
        feats_raw = m["teacher"](image)         # (1,C,H,W) -- teacher=384 ch
    feats_raw = feats_raw[0]                    # (C,H,W)
    feats_n = (feats_raw - n["t_mean"][:,None,None]) / n["t_std"][:,None,None]
    C,H,W = feats_n.shape
    feats_np = feats_n.cpu().float().numpy().reshape(C, H*W).T  # (H*W,C)
    from PIL import Image as PILImg
    comp_small = np.array(PILImg.fromarray(comp.astype(np.uint8))
                          .resize((W,H), PILImg.NEAREST))
    heat_small = np.zeros((H*W,), dtype=np.float32)
    n_parts    = fv[0].shape[0]
    for part_id in range(n_parts):
        mask = (comp_small.ravel() == part_id)
        if mask.sum() == 0:
            continue
        mu   = fv[0][part_id]
        cinv = fv[1][part_id]
        diff = feats_np[mask] - mu[None,:]
        d2   = np.sum(diff @ cinv * diff, axis=1)
        heat_small[mask] = np.sqrt(np.clip(d2, 0, None))
    heat_small = heat_small.reshape(H, W)
    heat256 = np.array(PILImg.fromarray(heat_small).resize((256,256), PILImg.BILINEAR))
    return heat256


def run_one(idx, test_set, models, norms, device, maps_root=None, category=None):
    m, n = models, norms
    image, seg, _, path = test_set[idx]
    image = image.unsqueeze(0)
    if ON_GPU: image, seg = image.cuda(), seg.cuda()

    # appearance branch
    with torch.no_grad():
        teacher_feats = m["teacher"](image)
        student_feats = m["student"](image)
        ae_out_raw    = m["autoencoder"](image)

    a_a_raw,_,_ = predict(image=image, teacher=m["teacher"], student=m["student"],
                           autoencoder=m["autoencoder"], teacher_mean=n["t_mean"],
                           teacher_std=n["t_std"],
                           q_st_start=n["q_st_s"], q_st_end=n["q_st_e"],
                           q_ae_start=n["q_ae_s"], q_ae_end=n["q_ae_e"])
    a_a = ((a_a_raw - n["q_eff_s"]) / n["q_eff_e"])[0,0].cpu().numpy()

    # composition branch
    a_c_raw = predict_comp_map(seg, m["comp_ae"], m["comp_unet"]).unsqueeze(0)
    a_c = ((a_c_raw - n["q_seg_s"]) / n["q_seg_e"])[0,0].cpu().numpy()

    # global branch
    fv = n["fv"]
    s_glob = predict_mahalanobis(
        image=image, seg=seg, teacher=m["student"], teacher_mean=n["t_mean"],
        feature_vectors_covinv=fv[1], feature_vectors_mean=fv[0],
        feature_vectors_covinv_seg=fv[3], feature_vectors_mean_seg=fv[2],
        feature_vectors_covinv_seg_area=fv[5], feature_vectors_mean_seg_area=fv[4],
        teacher_std=n["t_std"], q_start=n["q_mah_s"], q_end=n["q_mah_e"])

    comp = seg.argmax(0).cpu().numpy()
    with torch.no_grad():
        c_rec = m["comp_ae"](seg.unsqueeze(0)).softmax(dim=1).argmax(dim=1)[0].cpu().numpy()

    s_app  = float(np.max(a_a))
    s_comp = float(np.max(a_c))
    score  = s_app + s_comp + float(s_glob)
    typ    = os.path.basename(os.path.dirname(path))

    img_pil = Image.open(path).convert("RGB").resize((256,256))
    fg_mask     = sam_foreground(img_pil, device)
    _, dino_rgb = dino_clusters(img_pil, device, k=6)

    pseudo = None
    if maps_root and category:
        pseudo = pseudo_label_map(maps_root, category, path)

    teacher_rgb = feats_to_rgb(teacher_feats[0])
    student_rgb = feats_to_rgb(student_feats[0])

    from sklearn.decomposition import PCA
    from PIL import Image as PILImg

    def _to_flat(t):
        arr = t.cpu().float().numpy()
        C,H,W = arr.shape
        return arr.reshape(C, H*W).T, H, W

    tf, Ht, Wt = _to_flat(teacher_feats[0])
    sf, Hs, Ws = _to_flat(student_feats[0])
    pt = PCA(n_components=3).fit_transform(tf)
    ps = PCA(n_components=3).fit_transform(sf)

    def _norm01(x):
        lo,hi = x.min(0), x.max(0)
        return (x - lo) / (hi - lo + 1e-8)

    pt_n, ps_n = _norm01(pt), _norm01(ps)
    diff_small = np.linalg.norm(pt_n - ps_n, axis=1).reshape(Ht, Wt)
    diff256 = np.array(PILImg.fromarray(diff_small.astype(np.float32))
                       .resize((256,256), PILImg.BILINEAR))

    try:
        ae_rgb256 = feats_to_rgb(ae_out_raw[0])
    except Exception as e:
        print(f"  ae_rgb256 fallback: {e}")
        ae_rgb256 = np.array(Image.open(path).convert("RGB").resize((256,256)))

    try:
        mah_heat = mahal_heatmap(image, seg, models, norms, device)
    except Exception as e:
        print(f"  mahal_heatmap skipped: {e}")
        mah_heat = np.zeros((256,256), dtype=np.float32)

    return dict(path=path, a_a=a_a, a_c=a_c, comp=comp, c_rec=c_rec,
                score=score, s_app=s_app, s_comp=s_comp, s_glob=float(s_glob),
                type=typ, img_idx=idx+1, total=len(test_set),
                fg_mask=fg_mask, dino_rgb=dino_rgb, pseudo=pseudo,
                teacher_rgb=teacher_rgb, student_rgb=student_rgb,
                diff256=diff256, ae_rgb256=ae_rgb256, mah_heat=mah_heat)


def pseudo_label_map(maps_root, category, path):
    typ  = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    p = os.path.join(maps_root, category, "test", typ, stem + ".png")
    found = "test"
    if not os.path.exists(p):
        train_good = sorted(glob.glob(os.path.join(maps_root, category, "train", "good", "*.png")))
        if train_good:
            p = train_good[0]
            found = "train_fallback"
        else:
            return None
    raw = np.array(Image.open(p))
    if raw.ndim == 3:
        arr = raw[:,:,0]
    else:
        arr = raw
    arr = np.array(Image.fromarray(arr).resize((256, 256), Image.NEAREST))
    print(f"  pseudo-label ({found}): min={arr.min()} max={arr.max()} unique={np.unique(arr)[:8]}")
    return arr


def gt_mask(loco, category, path):
    typ  = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    mask = np.zeros((256,256), dtype=np.float32)
    for f in glob.glob(os.path.join(loco, category, "ground_truth", typ, stem, "*.png")):
        mask = np.maximum(mask,
               np.array(Image.open(f).convert("L").resize((256,256),Image.NEAREST)) > 0)
    return mask


def best_f1_threshold(all_scores, all_labels):
    scores = np.array(all_scores); labels = np.array(all_labels)
    best_t, best_f1 = sorted(scores)[0], 0.0
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        tp = np.sum((pred==1)&(labels==1))
        fp = np.sum((pred==1)&(labels==0))
        fn = np.sum((pred==0)&(labels==1))
        f1 = (2*tp)/(2*tp+fp+fn+1e-9)
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t, best_f1


# -- selection ----------------------------------------------------------------

def select_images(test_set, all_results, mode, threshold):
    by_type = {t:[] for t in TYPES}
    for i,(p,_) in enumerate(test_set.dataset1.samples):
        by_type[os.path.basename(os.path.dirname(p))].append(i)

    if mode == "random":
        rng = random.Random(42)
        chosen = []
        for t in TYPES:
            chosen += [all_results[i] for i in rng.sample(by_type[t], min(4,len(by_type[t])))]
        return chosen[:10]

    if mode == "best":
        good = sorted([all_results[i] for i in by_type["good"]], key=lambda r: r["score"])[:2]
        log  = sorted([all_results[i] for i in by_type["logical_anomalies"]], key=lambda r: -r["score"])[:4]
        st   = sorted([all_results[i] for i in by_type["structural_anomalies"]], key=lambda r: -r["score"])[:4]
        return good+log+st

    good = sorted([all_results[i] for i in by_type["good"]], key=lambda r: r["score"])[:2]
    log_all = sorted([all_results[i] for i in by_type["logical_anomalies"]], key=lambda r: -r["score"])
    str_all = sorted([all_results[i] for i in by_type["structural_anomalies"]], key=lambda r: -r["score"])
    log_det  = [r for r in log_all if r["score"]>=threshold][:3]
    log_miss = [r for r in log_all if r["score"]< threshold][-1:]
    str_det  = [r for r in str_all if r["score"]>=threshold][:3]
    str_miss = [r for r in str_all if r["score"]< threshold][-1:]
    if len(log_det) <3: log_det  = log_all[:3]
    if len(log_miss)<1: log_miss = [log_all[-1]] if log_all else []
    if len(str_det) <3: str_det  = str_all[:3]
    if len(str_miss)<1: str_miss = [str_all[-1]] if str_all else []
    return good+log_det+log_miss+str_det+str_miss


# -- figure helpers -----------------------------------------------------------

def fig_to_array(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h,w,3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h,w,4)[:,:,:3]


def save_panel(arr, cmap, path, title, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.axis("off")
    if cmap is None:
        ax.imshow(arr)
    elif cmap is COMP_PAL:
        ax.imshow(arr, cmap=COMP_PAL, vmin=0, vmax=9, interpolation="nearest")
    elif cmap == "gray":
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=4)
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(r, gt, img_rgb, out_path, category, threshold):
    detected = r["score"] >= threshold
    if r["type"] == "good":
        status = "NORMAL (correct)" if not detected else "FALSE POSITIVE"
    else:
        status = "DETECTED" if detected else "MISSED"

    fusion = r["a_a"] + r["a_c"]
    vmax   = max(float(np.percentile(np.abs(fusion), 99)), 1e-6)

    fig_ov, ax_ov = plt.subplots(figsize=(3,3))
    ax_ov.imshow(img_rgb)
    ax_ov.imshow(np.clip(fusion,0,None), cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    ax_ov.axis("off")
    overlay_arr = fig_to_array(fig_ov)
    plt.close(fig_ov)

    fg_vis = img_rgb.copy().astype(float)
    fg_vis[~r["fg_mask"]] *= 0.25
    fg_vis = fg_vis.clip(0,255).astype(np.uint8)

    pseudo_vis = r.get("pseudo")
    if pseudo_vis is None or (pseudo_vis.max() == 0):
        pseudo_vis = r["comp"].copy().astype(np.uint8)

    diff_v  = r.get("diff256",  np.zeros((256,256),dtype=np.float32))
    mah_v   = r.get("mah_heat", np.zeros((256,256),dtype=np.float32))
    dv_max  = max(float(diff_v.max()), 1e-6)
    mv_max  = max(float(mah_v.max()),  1e-6)

    panels = [
        (img_rgb,                  None,     "1. RGB input"),
        (fg_vis,                   None,     "2. SAM foreground\n(background dimmed)"),
        (r["dino_rgb"],            None,     "3. DINO clusters\n(K=6 ViT-B/8)"),
        (pseudo_vis,               COMP_PAL, "4. Pseudo-labels\n(majority vote)"),
        (r["comp"],                COMP_PAL, "5. Composition C\n(UNet output)"),
        (r["c_rec"],               COMP_PAL, "6. Normal C_rec\n(AE reconstruction)"),
        (r.get("teacher_rgb", img_rgb), None, "7. Teacher features\n(PCA->RGB)"),
        (r.get("student_rgb", img_rgb), None, "8. Student features\n(mimics teacher)"),
        (diff_v,                   "hot",    "9. Teacher-Student diff\n(raw A_a signal)"),
        (r.get("ae_rgb256", img_rgb), None,  "10. AE reconstruction\n(normal appearance)"),
        (np.clip(r["a_a"],0,None), "magma",  f"11. Appearance A_a\n(peak={r['s_app']:.3f})"),
        (np.clip(r["a_c"],0,None), "magma",  f"12. Composition A_c\n(peak={r['s_comp']:.3f})"),
        (mah_v,                    "plasma", f"13. Mahal. heatmap\n(S_g={r['s_glob']:.3f})"),
        (overlay_arr,              None,     "14. Fused overlay\n(anomaly on RGB)"),
        (gt,                       "gray",   "15. Ground truth"),
    ]

    fig = plt.figure(figsize=(60, 6.0))
    gs  = gridspec.GridSpec(1, len(panels), figure=fig, wspace=0.04)
    for k,(arr,cm,ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if cm is None:
            ax.imshow(arr)
        elif cm is COMP_PAL:
            ax.imshow(arr, cmap=COMP_PAL, vmin=0, vmax=9, interpolation="nearest")
        elif cm == "gray":
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            pv = float(np.percentile(arr, 99)) if arr.max() > 0 else 1.0
            ax.imshow(arr, cmap=cm, vmin=0, vmax=max(pv, 1e-6))
        ax.set_title(ttl, fontsize=10, pad=4, loc="center")

    typ_label = {"good":"good","logical_anomalies":"logical",
                 "structural_anomalies":"structural"}[r["type"]]
    suptitle = (
        f"{category}  |  test image #{r['img_idx']} of {r['total']}  "
        f"|  type: {typ_label}  |  threshold: {threshold:.3f}  ->  {status}\n"
        f"Appearance max(A_a): {r['s_app']:.3f}     "
        f"Composition max(A_c): {r['s_comp']:.3f}     "
        f"Global Mahalanobis S_g: {r['s_glob']:.3f}     "
        f"Fused total: {r['score']:.3f}"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.04)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_info_txt(r, gt, threshold, out_path):
    detected  = r["score"] >= threshold
    anomalous = r["type"] != "good"
    if not anomalous:
        outcome = "correct normal" if not detected else "false positive"
    else:
        outcome = "DETECTED" if detected else "MISSED"

    fusion   = r["a_a"] + r["a_c"]
    hot      = np.unravel_index(np.argmax(np.abs(fusion)), fusion.shape)
    py, px   = hot
    part_id  = int(r["comp"][py,px])
    prec_id  = int(r["c_rec"][py,px])
    fg_pct   = 100*r["fg_mask"].sum()/r["fg_mask"].size

    lines = [
        f"Category : {os.path.basename(os.path.dirname(os.path.dirname(r['path'])))}",
        f"Image    : test image #{r['img_idx']} of {r['total']}",
        f"File     : {r['path']}",
        f"Type     : {r['type']}",
        "",
        "-- Scores --",
        f"  Appearance  max(A_a)   : {r['s_app']:.6f}",
        f"  Composition max(A_c)   : {r['s_comp']:.6f}",
        f"  Global Mahalanobis S_g : {r['s_glob']:.6f}",
        f"  Fused total score      : {r['score']:.6f}",
        f"  Threshold (best-F1)    : {threshold:.6f}",
        f"  Decision               : {outcome.upper()}",
        "",
        "-- SAM foreground --",
        f"  Foreground pixels      : {fg_pct:.1f}% of image",
        "",
        "-- DINO clusters --",
        "  Model: ViT-B/8 pretrained DINO",
        "  Patch grid: 28x28 (for 224x224 input)",
        "  K=6 KMeans clusters on patch features",
        "",
        "-- Pixel walkthrough (hottest pixel) --",
        f"  Pixel location (row, col) : ({py}, {px})",
        f"  Composition C  part ID    : {part_id}  ({PART_NAMES.get(part_id,'?')})",
        f"  Normal C_rec   part ID    : {prec_id}  ({PART_NAMES.get(prec_id,'?')})",
        f"  Composition changed?      : {'YES' if part_id!=prec_id else 'no'}",
        f"  A_a value at hot pixel    : {r['a_a'][py,px]:.6f}",
        f"  A_c value at hot pixel    : {r['a_c'][py,px]:.6f}",
        f"  Fused value at hot pixel  : {fusion[py,px]:.6f}",
    ]
    Path(out_path).write_text("\n".join(lines))


# -- main per-category --------------------------------------------------------

def process_category(args, category):
    loco   = os.path.expanduser(args.loco)
    maps   = os.path.expanduser(args.maps)
    res    = os.path.expanduser(args.results)
    out    = os.path.expanduser(args.out)
    device = torch.device("cuda" if ON_GPU else "cpu")

    print(f"\n{'='*60}\nCategory: {category}\n{'='*60}")
    models = load_models(res, category)
    _, val_loader, train_loader, test_set = build_loaders(loco, maps, category)
    norms  = normalise(res, category, models, train_loader, val_loader)
    N      = len(test_set)
    print(f"Test set: {N} images")

    print("Computing scores + SAM/DINO for all test images...")
    all_results = []
    for i in range(N):
        print(f"  {i+1}/{N}", end="\r", flush=True)
        all_results.append(run_one(i, test_set, models, norms, device,
                                   maps_root=maps, category=category))
    print()

    all_scores = [r["score"] for r in all_results]
    all_labels = [0 if r["type"]=="good" else 1 for r in all_results]
    threshold, best_f1 = best_f1_threshold(all_scores, all_labels)
    print(f"Threshold (best-F1={best_f1:.3f}): {threshold:.4f}")

    chosen = select_images(test_set, all_results, args.select, threshold)
    print(f"Selected {len(chosen)} images (mode={args.select})")

    cat_out = os.path.join(out, category)
    os.makedirs(cat_out, exist_ok=True)

    for rank, r in enumerate(chosen, 1):
        typ       = r["type"]
        fname     = os.path.splitext(os.path.basename(r["path"]))[0]
        typ_label = {"good":"good","logical_anomalies":"log","structural_anomalies":"str"}[typ]
        detected  = r["score"] >= threshold
        if typ == "good":
            det_label = "normal"
        else:
            det_label = "detected" if detected else "missed"

        folder_name = f"{rank:02d}_{typ_label}_{det_label}_img{r['img_idx']:03d}_{fname}"
        img_dir = os.path.join(cat_out, folder_name)
        os.makedirs(img_dir, exist_ok=True)

        img_rgb = np.array(Image.open(r["path"]).convert("RGB").resize((256,256)))
        gt      = gt_mask(loco, category, r["path"])
        fusion  = r["a_a"] + r["a_c"]
        vmax    = max(float(np.percentile(np.abs(fusion), 99)), 1e-6)

        fg_vis = img_rgb.copy().astype(float)
        fg_vis[~r["fg_mask"]] *= 0.25
        fg_vis = fg_vis.clip(0,255).astype(np.uint8)

        fig_ov, ax_ov = plt.subplots(figsize=(3,3))
        ax_ov.imshow(img_rgb)
        ax_ov.imshow(np.clip(fusion,0,None), cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        ax_ov.axis("off")
        overlay_arr = fig_to_array(fig_ov)
        plt.close(fig_ov)

        pseudo_vis = r.get("pseudo")
        if pseudo_vis is None:
            pseudo_vis = np.zeros((256,256), dtype=np.uint8)

        diff_v = r.get("diff256",  np.zeros((256,256),dtype=np.float32))
        mah_v  = r.get("mah_heat", np.zeros((256,256),dtype=np.float32))
        dv_max = max(float(diff_v.max()), 1e-6)
        mv_max = max(float(mah_v.max()),  1e-6)

        save_panel(img_rgb,  None,     os.path.join(img_dir,"01_rgb.png"),
                   f"1. RGB input\ntest image #{r['img_idx']} of {r['total']}")
        save_panel(fg_vis,   None,     os.path.join(img_dir,"02_sam_foreground.png"),
                   "2. SAM foreground\n(background dimmed)")
        save_panel(r["dino_rgb"], None, os.path.join(img_dir,"03_dino_clusters.png"),
                   "3. DINO clusters\n(K=6 on ViT-B/8 patches)")
        save_panel(pseudo_vis, COMP_PAL, os.path.join(img_dir,"04_pseudo_labels.png"),
                   "4. Pseudo-labels\n(majority vote)")
        save_panel(r["comp"], COMP_PAL, os.path.join(img_dir,"05_composition_C.png"),
                   "5. Composition map C\n(UNet part labels)")
        save_panel(r["c_rec"],COMP_PAL, os.path.join(img_dir,"06_composition_Crec.png"),
                   "6. Normal C_rec\n(AE reconstruction)")
        save_panel(r.get("teacher_rgb", img_rgb), None,
                   os.path.join(img_dir,"07_teacher_features.png"),
                   "7. Teacher features\n(PCA->RGB)")
        save_panel(r.get("student_rgb", img_rgb), None,
                   os.path.join(img_dir,"08_student_features.png"),
                   "8. Student features\n(mimics teacher)")
        save_panel(diff_v, "hot",
                   os.path.join(img_dir,"09_teacher_student_diff.png"),
                   "9. Teacher-Student diff\n(raw A_a signal)", vmin=0, vmax=dv_max)
        save_panel(r.get("ae_rgb256", img_rgb), None,
                   os.path.join(img_dir,"10_ae_reconstruction.png"),
                   "10. AE reconstruction\n(normal appearance)")
        save_panel(np.clip(r["a_a"],0,None),"magma",
                   os.path.join(img_dir,"11_appearance_Aa.png"),
                   f"11. Appearance A_a\nmax={r['s_app']:.3f}", vmin=0, vmax=vmax)
        save_panel(np.clip(r["a_c"],0,None),"magma",
                   os.path.join(img_dir,"12_composition_Ac.png"),
                   f"12. Composition A_c\nmax={r['s_comp']:.3f}", vmin=0, vmax=vmax)
        save_panel(mah_v, "plasma",
                   os.path.join(img_dir,"13_mahal_heatmap.png"),
                   f"13. Mahal. heatmap\nS_g={r['s_glob']:.3f}", vmin=0, vmax=mv_max)
        save_panel(overlay_arr, None,
                   os.path.join(img_dir,"14_fused_overlay.png"),
                   "14. Fused overlay\n(anomaly on RGB)")
        save_panel(gt,"gray",
                   os.path.join(img_dir,"15_groundtruth.png"),
                   "15. Ground truth mask")

        save_summary(r, gt, img_rgb,
                     os.path.join(img_dir,"16_summary_all_steps.png"),
                     category, threshold)
        save_info_txt(r, gt, threshold, os.path.join(img_dir,"info.txt"))

        print(f"  [{rank:02d}] #{r['img_idx']:03d}/{r['total']} {typ_label} {det_label} "
              f"score={r['score']:.3f} -> {folder_name}")

    print(f"\nDone. Output: {cat_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="juice_bottle",
                    choices=CATS+["all"])
    ap.add_argument("--results", default="~/salad/output/results")
    ap.add_argument("--loco",    default="~/salad/SALAD/data/mvtec_loco")
    ap.add_argument("--maps",    default="~/salad/SALAD/data/mvtec_loco_composition_maps")
    ap.add_argument("--out",     default="~/salad/output/walkthrough")
    ap.add_argument("--select",  default="mixed",
                    choices=["mixed","best","random"])
    args = ap.parse_args()

    cats = CATS if args.category=="all" else [args.category]
    for c in cats:
        d = os.path.join(os.path.expanduser(args.results), c)
        if not os.path.exists(os.path.join(d,"comp_unet_final.pth")):
            print(f"SKIP {c}: model not found")
            continue
        process_category(args, c)


if __name__ == "__main__":
    main()
