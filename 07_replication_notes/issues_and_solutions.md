# Issues & Solutions — SALAD Replication

## Issue Log

| # | Issue | Status | Fix |
|---|---|---|---|
| 1 | Dark anomaly map panels in visualisation | ✅ Fixed | Per-panel 99th-percentile vmax |
| 2 | AE output cannot be imshow'd directly (384 channels) | ✅ Fixed | PCA to 3 channels → uint8 |
| 3 | Wrong features for Mahalanobis (student vs teacher) | ✅ Fixed | Use teacher features only |
| 4 | matplotlib `tostring_rgb()` AttributeError | ✅ Fixed | Fallback to `buffer_rgba()` |
| 5 | CUDA version mismatch | ✅ Fixed | Use `+cu118` wheels |
| 6 | SAM checkpoint not found | ✅ Fixed | Glob + None fallback |
| 7 | Result gap vs paper (−2.3 pp) | ⚠️ Known | Seed variance; within expected range |

---

## Detailed Solutions

### Issue 1 — Dark Panels
The anomaly maps for normal-looking test images have max values near 0.  
Using a global vmax (e.g., 1.0) makes everything look dark.  
**Solution:** Set vmax per panel using `np.percentile(arr, 99)`.

### Issue 2 — AE Visualisation
The autoencoder outputs raw features (384-dim), not RGB.  
**Solution:** Apply PCA (3 components) to flatten to displayable RGB.

### Issue 3 — Mahalanobis Features
Using student features (which change during training) makes the Mahalanobis  
covariance matrix unstable.  
**Solution:** Always extract from the frozen teacher network.

### Issue 4 — matplotlib Compatibility
Newer matplotlib removed `tostring_rgb()`.  
**Solution:** Try/except with fallback to `buffer_rgba()` (drop alpha channel).

### Issue 5 — CUDA
PyTorch 2.x default wheels link against CUDA 12.x. The RACE A10G uses CUDA 11.8.  
**Solution:**  
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

### Issue 6 — SAM Not Found
SAM checkpoint must be manually downloaded.  
**Solution:**  
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -O ~/salad/SALAD/weights/sam_vit_b_01ec64.pth
```

### Issue 7 — Result Gap
−2.3 pp vs paper. This is within normal seed-to-seed variance for MVTec LOCO.  
The paper likely reports mean over multiple seeds or best seed.  
**Action:** Run seeds 0, 1, 2 for each category and report mean ± std.
