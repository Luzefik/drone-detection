# Drone Detection Using Principal Component Analysis

**Course:** Linear Algebra
**Authors:** Marko Pasternak, Nestor Leyko, Denys Marchenko, Andrii Kulbaba
**Date:** March 2026

---

## Video explanations

[Marko Pasternak](https://youtu.be/oazqTDouZgw)

[Andrii Kulbaba](https://youtu.be/JVAYNW9Tp7A?si=cT0A8HeXPGJ-eHVO)

[Nestor Leyko](https://youtu.be/pd_xpwEbOXQ)

[Denys Marchenko](https://youtu.be/V6JO1SLM50c?si=GsLyUgEOPpOJx7cF)

---

## Abstract

This project applies Principal Component Analysis (PCA) to the problem of detecting First Person View (FPV) drones in images. By constructing an "Eigendrone" subspace from a set of training images, the system projects unseen test images into this lower-dimensional space and classifies them based on Euclidean distance. The entire pipeline is implemented from scratch using core linear algebra operations — eigenvalue decomposition, covariance computation, and orthogonal projection — without reliance on high-level machine learning frameworks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Algorithm](#algorithm)
4. [Project Structure](#project-structure)
5. [Installation and Usage](#installation-and-usage)
6. [Anticipated Challenges](#anticipated-challenges)
7. [Project Timeline](#project-timeline)
8. [References](#references)

---

## Introduction

Image classification is a fundamental problem in computer vision. While modern approaches rely on deep neural networks, classical methods rooted in linear algebra remain valuable both for their interpretability and as pedagogical tools.

In this project, we build a PCA-based detection pipeline that:

1. Represents each image as a high-dimensional vector.
2. Learns a compact basis (the "Eigendrone" space) that captures the principal modes of variation across drone images.
3. Classifies a test image by projecting it into this space and measuring its distance to known drone projections.

This approach directly applies the following linear algebra concepts: matrix decomposition, eigenvalue problems, the Power Method, and orthogonal projection.

---

## Mathematical Background

### Image Representation

Each RGB image is converted to grayscale and flattened from an N × M matrix into a column vector of dimension D = N·M. The training set is assembled into a data matrix:

$$X = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^{D \times n} \tag{1}$$

### Data Centering

The mean vector is computed and subtracted to center the data at the origin:

$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i \tag{2}$$

$$\bar{X} = X - \mu \tag{3}$$

### Covariance Matrix

The covariance matrix is defined as:

$$C = \frac{1}{n} \bar{X}\bar{X}^T \in \mathbb{R}^{D \times D} \tag{4}$$

Since D >> n for high-resolution images, we instead compute the surrogate matrix:

$$C' = \bar{X}^T \bar{X} \in \mathbb{R}^{n \times n} \tag{5}$$

If v is an eigenvector of C', then the corresponding eigenvector of C is recovered as:

$$(\bar{X}\bar{X}^T)(\bar{X}v) = \lambda(\bar{X}v) \tag{6}$$

### Power Method

The dominant eigenvector is found iteratively:

$$x_{k+1} = \frac{Ax_k}{\|Ax_k\|} \tag{7}$$

This process is repeated for the top K principal components, which form the projection basis Y.

### Projection and Classification

Training images are projected into the eigendrone space:

$$Z = Y^T \bar{X} \tag{8}$$

A test image is centered and projected:

$$Z_{\text{test}} = Y^T(x_{\text{test}} - \mu) \tag{9}$$

Classification is performed by computing the minimum Euclidean distance to training projections:

$$\epsilon = \min_i \|Z_{\text{test}} - Z_i\|$$

If ε < τ (a predefined threshold), the image is classified as containing a drone.

---

## Algorithm

```
Algorithm: PCA Training and Drone Detection (Eigendrone Method)

Input:  Training images I_1, ..., I_N (size w × h)
        Test image x_test
        Threshold τ

Phase 1 — Training:
  1. Flatten each image I_i into a column vector x_i ∈ R^D, where D = w × h
  2. Compute mean vector: μ = (1/N) Σ x_i
  3. Center data: X̄_i = x_i − μ
  4. Construct data matrix: A = [X̄_1, ..., X̄_N]
  5. Compute surrogate covariance: C' = A^T A
  6. For j = 1 to K:
       Use the Power Method to find dominant eigenvector v_j of C'
       Compute eigendrone: y_j = A v_j
       Normalize: y_j = y_j / ||y_j||
  7. Form projection basis: Y = [y_1, ..., y_K]
  8. Project training data: Z_i = Y^T X̄_i

Phase 2 — Detection:
  1. Center test image: ζ = x_test − μ
  2. Project: Z_test = Y^T ζ
  3. Compute distance: ε = min_i ||Z_test − Z_i||
  4. If ε < τ → "Drone Detected", else → "No Drone"
```

---

## Project Structure

```
drone-detection/
├── detect.ipynb              — Main notebook: full PCA pipeline implementation
├── requirements.txt          — Python dependencies
├── example-test.jpg          — Sample test image for evaluation
├── First_interim_report.pdf  — Interim project report
├── LA_Video_Instruction.pdf  — Mathematical methodology documentation
├── LICENSE                   — MIT License
└── README.md
```

---

## Installation and Usage

### Requirements

- Python 3.10 or higher

### Setup

```bash
git clone https://github.com/<username>/drone-detection.git
cd drone-detection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

| Package          | Version  | Purpose                                |
|------------------|----------|----------------------------------------|
| `numpy`          | ≥ 1.26.0 | Matrix operations, eigenvalue computation |
| `matplotlib`     | ≥ 3.7.1  | Visualization                          |
| `seaborn`        | ≥ 0.12.2 | Statistical plotting                   |
| `opencv-python`  | ≥ 4.8.0  | Image I/O and grayscale conversion     |

### Running

```bash
jupyter notebook detect.ipynb
```

---

## Anticipated Challenges

| Challenge | Description |
|-----------|-------------|
| Background variation | PCA may capture environment noise rather than drone features when backgrounds differ significantly across the dataset. |
| Scale and rotation invariance | PCA does not inherently handle changes in scale or orientation. Dataset preprocessing (cropping, alignment) is required. |
| Computational constraints | While the surrogate matrix C' reduces the problem size, eigenvalue computation for very large datasets remains memory-intensive. |

---

## Project Timeline

| Phase | Period | Deliverables |
|-------|--------|-------------|
| 1 | Until 11.03 | Literature review and mathematical derivation of PCA |
| 2 | Until 25.03 | Data acquisition: FPV imagery collection and grayscale preprocessing |
| 3 | 20.03 – 05.04 | Algorithm development: Power Method implementation |
| 4 | Final | Testing, validation, threshold adjustment, and final report |

---

## References

1. Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
2. Starmer, J. *StatQuest: Principal Component Analysis (PCA) Step-by-Step*. [YouTube](https://www.youtube.com/watch?v=FgakZw6K1QQ).
3. Brunton, S. *Singular Value Decomposition (SVD) Series*. [YouTube](https://www.youtube.com/watch?v=gXbThCXjZFM).
4. Sanderson, G. (3Blue1Brown). *Essence of Linear Algebra*. [YouTube](https://www.youtube.com/watch?v=PFDu9oVAE-g).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
