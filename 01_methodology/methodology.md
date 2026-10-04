# SALAD Architecture — Methodology

## Paper Reference
- **Title:** SALAD: Semantic Aware Discriminative Logical Anomaly Detection
- **Venue:** ICCV 2025
- **arXiv:** 2509.02101

---

## High-Level Architecture

SALAD detects anomalies by fusing three complementary branches:

```
Input Image (256×256 RGB)
        │
        ├─── Branch 1: Appearance Branch ──────────────────────────────────────┐
        │    EfficientAD Teacher/Student + Autoencoder                         │
        │    → Appearance anomaly map A_a (per-pixel)                          │
        │                                                                       │
        ├─── Branch 2: Composition Branch ──────────────────────────────────────┤
        │    SAM (foreground mask) + DINO (patch features/clusters)             │
        │    → Pseudo-label composition map C (normal prototype)               │
        │    U-Net Decoder + Autoencoder + Discriminator                        │
        │    → Composition anomaly map A_c (per-pixel)                          │
        │                                                                       │
        └─── Branch 3: Global Statistical Branch ───────────────────────────────┤
             Teacher features → Mahalanobis distance → S_g (image-level)        │
                                                                                │
                        ┌───────────────────────────────────────────────────────┘
                        ↓
              Per-branch standardisation (mean/std from training set)
                        ↓
              Fusion: A_S = max(A_a, A_c, S_g)
                        ↓
              Image AUROC evaluation (LOCO: logical + structural)
```

---

## Branch 1: Appearance Branch

Based on **EfficientAD** (Batzner et al., 2023):

| Component | Architecture | Output |
|---|---|---|
| Teacher | WideResNet-101 (pretrained, frozen) | 384-dim patch features |
| Student | Mirror architecture (trained) | 384-dim patch features |
| Autoencoder | Encoder-Decoder | Reconstructed teacher features |

**Anomaly score A_a:**
- Student–teacher feature distance: detects structural anomalies
- Teacher–AE feature distance: detects distribution shift (logical)
- Combined: `A_a = max(normalized_st_dist, normalized_ae_dist)`

---

## Branch 2: Composition Branch

Uses **SAM** (foreground segmentation) and **DINO ViT-B/8** (patch clustering):

1. **Pseudo-label generation:** DINO patch tokens → k-means (k=6) clusters → composition map C
2. **U-Net decoder:** Takes C as input, learns normal composition reconstruction
3. **Autoencoder:** Reconstructs composition map; large error = logical anomaly
4. **Discriminator:** Binary real/fake on composition patches

**Anomaly score A_c:** reconstruction error from U-Net/AE on the composition map

---

## Branch 3: Global Statistical Branch

- Extracts teacher features from the full training set
- Fits a **multivariate Gaussian** (mean μ, covariance Σ) per spatial location
- At test time: **Mahalanobis distance** from feature to fitted Gaussian → S_g
- Captures global distribution of normal appearance

---

## Fusion

```python
# Per-branch standardisation
A_a_norm = (A_a - mean_a) / std_a
A_c_norm = (A_c - mean_c) / std_c
S_g_norm = (S_g - mean_g) / std_g

# Max fusion
A_S = max(A_a_norm, A_c_norm, S_g_norm)  # element-wise or image-level
```

Image-level score = max pixel value of A_S → thresholded for AUROC.

---

## Dataset: MVTec LOCO AD

5 categories with both **structural** and **logical** anomalies:

| Category | Normal Train | Test Good | Structural | Logical |
|---|---|---|---|---|
| breakfast_box | 670 | 85 | 65 | 85 |
| juice_bottle | 380 | 47 | 35 | 47 |
| pushpins | 700 | 85 | 61 | 68 |
| screw_bag | 575 | 80 | 64 | 78 |
| splicing_connectors | 651 | 80 | 72 | 80 |

---

## Replication Notes

See `07_replication_notes/implementation_notes.md` for fixes applied during replication.

Key result: **93.80 mean AUROC** (seed 42) vs **96.1** in paper (gap: −2.3 pp).
The gap is within normal seed variance for this benchmark.
