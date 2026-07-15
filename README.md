# Zero-Dependency Visual Difference Detector & Image Aligner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy Only](https://img.shields.io/badge/dependencies-pure%20numpy-brightgreen.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-precision, **zero-dependency** computer vision pipeline engineered from scratch for automated visual anomaly detection and geometric alignment (affine warping).

Unlike standard implementations that rely on heavy CV frameworks like OpenCV, scikit-image, or SciPy, **this entire project is built exclusively using Python and pure NumPy**. Every core algorithm — from alignment marker segmentation and Ordinary Least Squares (OLS) affine transformation matrices to fully vectorized bilinear interpolation and morphological noise suppression — is implemented mathematically from the ground up.

---

## Pipeline Visualization

The pipeline processes distorted, unaligned input images, dynamically computes spatial transformations, and isolates pixel-perfect visual discrepancies with high tolerance for interpolation noise:

![Pipeline Visualization](docs/images/sample_000_pipeline.png)

---

## Algorithmic Architecture

The execution pipeline is decoupled into five sequential, highly optimized stages:

```text
[Distorted Input Image] 
         │
         ▼
1. Marker Segmentation (Quadrant-based Centroid Clustering)
         │
         ▼
2. Affine Transformation Matrix (Overdetermined OLS System: X @ M = Y)
         │
         ▼
3. Vectorized Bilinear Interpolation & Inverse Spatial Warping
         │
         ▼
4. Euclidean Norm Color Metric in RGB Space
         │
         ▼
5. Custom Morphological Erosion (4-Connectivity Noise Filtering)
         │
         ▼
[High-Precision Binary Difference Mask]


### 1. Geometric Alignment via Ordinary Least Squares (OLS)
To realign images subjected to translation, scaling, and shear, the algorithm locates four registration markers in the target grid. Rather than assuming a simple linear shift, we model the mapping as an affine transformation.

Given reference coordinates $X_{ref}$ and detected source centroids $Y_{src}$, we solve the overdetermined linear system using the Least Squares method:

$$X_{ref} \cdot M = Y_{src}$$

This yields the optimal transformation matrix $M$ that minimizes spatial squared error across all quadrants.

### 2. Fully Vectorized Bilinear Interpolation
To eliminate aliasing and jagged edges during grid transformation, we perform inverse spatial mapping coupled with custom bilinear interpolation. For every target coordinate $(x, y)$, the pixel intensity is computed from its four nearest neighbors in the source image:

$$I(x, y) = I_{00}w_a + I_{10}w_b + I_{01}w_c + I_{11}w_d$$

### 3. Euclidean RGB Distance Metric
Simple channel averaging (Mean Absolute Error) often fails to capture subtle, low-contrast anomalies. This pipeline implements the **Euclidean norm** across the RGB vector space to quantify perceptual color variance:

$$\text{Diff}(x, y) = \sqrt{\Delta R^2 + \Delta G^2 + \Delta B^2}$$

### 4. Custom Morphological Noise Suppression
Inverse warping inevitably introduces boundary interpolation artifacts ("halo noise"). To separate true structural differences from interpolation noise without using `cv2.erode`, we implemented a custom **4-connectivity morphological erosion operator**. A divergent pixel is classified as a true anomaly if and only if its structural divergence is confirmed by all four adjacent neighbors (top, bottom, left, and right):

$$\text{CleanMask}_{i, j} = \text{Mask}_{i, j} \land \text{Mask}_{i-1, j} \land \text{Mask}_{i+1, j} \land \text{Mask}_{i, j-1} \land \text{Mask}_{i, j+1}$$

---

## Performance 

The segmentation accuracy of the pipeline is evaluated using the industry-standard **Intersection over Union (IoU)** metric (Jaccard Index):

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

| Metric | Measured Performance | Notes |
| :--- | :--- | :--- |
| **Mean IoU Score** | **> 0.9500** | Evaluated across diverse geometric distortions |
| **Avg. Execution Time** | **~0.08–0.11 sec** | Tested on single-core CPU (**800 × 600** input) |
| **Memory Footprint** | **Minimal** | In-place array operations & vectorized memory views |
| **External CV Dependencies** | **0 (Zero)** | No OpenCV, SciPy, or scikit-image required |
