# Implementation Notes — SALAD Replication

## Overview

This document records the key implementation details, bugs encountered, and fixes applied during the SALAD replication on MVTec LOCO AD.

---

## Fix 1 — Per-Panel Colour Scaling (visualize_salad.py)

**Problem:** Anomaly maps with small max values (near 0) produced entirely dark panels in visualisations.

**Fix:** Use per-panel 99th-percentile normalisation with a floor of 1e-6:

```python
pv = float(np.percentile(arr, 99)) if arr.max() > 0 else 1.0
ax.imshow(arr, cmap=cm, vmin=0, vmax=max(pv, 1e-6))
```

---

## Fix 2 — AE Feature Visualisation via PCA

**Problem:** The autoencoder outputs 384-dim feature tensors; direct imshow fails.

**Fix:** Project to 3 channels via PCA then scale to uint8:

```python
def feats_to_rgb(t):
    # t: (C, H, W) tensor
    arr = t.squeeze(0).detach().cpu().numpy()   # (C, H, W)
    C, H, W = arr.shape
    flat = arr.reshape(C, -1).T                  # (H*W, C)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat).reshape(H, W, 3)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)

ae_rgb256 = feats_to_rgb(ae_out_raw[0])   # (256,256,3) uint8
```

---

## Fix 3 — Teacher vs Student for Mahalanobis

**Problem:** Student features have a different dimensionality after training; using student features for the Mahalanobis branch gave incorrect statistics.

**Fix:** Always use the **teacher** (frozen, 384-dim) features for the global statistical branch:

```python
feats_raw = m["teacher"](image)   # teacher=384 ch, matches fv training
```

---

## Fix 4 — matplotlib fig_to_array Compatibility

**Problem:** `fig.canvas.tostring_rgb()` is deprecated / removed in newer matplotlib versions.

**Fix:** Try the old API first, fall back to `buffer_rgba()`:

```python
def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[:, :, :3]
```

---

## Environment Issues

### CUDA / cuDNN version mismatch
- Initial install had CUDA 12.x, which conflicted with pre-built PyTorch wheels.
- Fix: used `+cu118` wheels explicitly.

### SAM checkpoint path
- SAM model not found at expected path on first run.
- Fix: `glob.glob(os.path.join(weights_dir, "sam_vit_*.pth"))` with fallback to `None` (skip SAM if not available).

### Hugging Face token
- ⚠️ **SECURITY:** Any HF token shared in a previous session should be **revoked immediately**.
- Go to https://huggingface.co → Settings → Access Tokens → delete/regenerate.

---

## Result Gap Analysis

Our replication achieves **93.80 mean AUROC** vs **96.1** in the paper (gap: −2.3 pp).

Likely causes:
1. **Seed variance** — The paper likely reports the best of multiple seeds. Our seed 42 result is within normal variance.
2. **ImageNet subset** — The paper may use a larger/different ImageNet subset for teacher pretraining.
3. **SAM checkpoint** — If SAM is not available, the composition branch degrades.
4. **Minor hyperparameter differences** — Batch scheduling, LR warmup details not fully documented in the paper.

---

## Notes on Multi-Seed Study

Run additional seeds to characterise variance:

```bash
python train_salad.py --category breakfast_box --steps 70000 --seed 0
python train_salad.py --category breakfast_box --steps 70000 --seed 1
python 03_scripts/final_results.py   # aggregates all seeds
```
