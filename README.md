# Perfusion Habitat Analysis for Diffuse Glioma

[![DOI](https://zenodo.org/badge/891887348.svg)](https://doi.org/10.5281/zenodo.21350691)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trained models and inference pipeline accompanying the manuscript **"Perfusion-based Habitat Analysis for Visualization and Quantification of Hemodynamic Heterogeneity in Diffuse Glioma"** (submitted to Cell Reports Medicine).

The materials here let a user apply the trained discrete variational autoencoder (dVAE) and the four Random Forest classifiers to their own DSC perfusion MRI, producing a voxel-wise habitat map, an 8-element Perfusion Habitat Ratio (PHR) vector, and probability outputs for WHO grade, IDH mutation, 1p/19q codeletion, and Ki-67 index.

**Citation** — this release is archived on Zenodo with version DOI [`10.5281/zenodo.21350691`](https://doi.org/10.5281/zenodo.21350691). The concept DOI [`10.5281/zenodo.21350692`](https://doi.org/10.5281/zenodo.21350692) always resolves to the latest release.

## 1. Layout

```
├── README.md                          this file
├── LICENSE                            MIT
├── requirements.txt                   pip dependencies
├── model/
│   ├── configs/vqgan_sep.yaml         dVAE architecture used to train the checkpoint
│   ├── taming/                        dVAE source (encoder / decoder / vector quantizer)
│   ├── encode.py                      voxel-wise inference; writes habitat NIfTI
│   ├── project_utils.py               instantiate_from_config helper
│   └── weights/
│       ├── dvae_checkpoint.ckpt       17 MB, trained dVAE (epoch 74)
│       └── rf/                        four Random Forest classifiers
└── analysis/
    └── predict.py                     habitat NIfTI + tumor mask → PHR → four endpoint probabilities
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.10–3.13 on Linux. GPU is optional; the deposited checkpoint runs on CPU for single-patient inference.

## 3. Inference pipeline

Given a preprocessed DSC 4D NIfTI (99.5th-percentile-normalised voxel time-series within a tumor mask) and a binary tumor mask NIfTI, the two steps below produce the habitat map, the PHR vector, and the four endpoint probabilities.

### Step 1 — Voxel-wise habitat map

```bash
cd model
python encode.py --dataroot <path/to/patient/root> \
       --resume weights/dvae_checkpoint.ckpt \
       -c configs/vqgan_sep.yaml
```

Writes `dsc_clusters/dVAE_quantization.nii.gz` in the patient directory. Voxels inside the tumor mask carry integer labels 1–8 corresponding to the eight Deep Pattern Habitats; voxels outside the mask are zero.

### Step 2 — PHR + classification probabilities

```bash
cd analysis
python predict.py \
       --habitat <path/to/dVAE_quantization.nii.gz> \
       --mask    <path/to/tumor_mask.nii.gz>
```

Prints the 8-element PHR (`H1..H8`) and, for each of the four classifiers, the class probabilities. Example output:

```
Perfusion Habitat Ratio (H1..H8):
  H1: 0.0210
  H2: 0.0483
  H3: 0.0972
  H4: 0.1204
  H5: 0.1855
  H6: 0.1620
  H7: 0.3410
  H8: 0.0246

Classification probabilities:
  who_grade: P(2)=0.021, P(3)=0.132, P(4)=0.847
  idh:       P(0)=0.912, P(1)=0.088
  1p19q:     P(0)=0.831, P(1)=0.169
  ki_67:     P(0)=0.213, P(1)=0.787
```

Class labels: `who_grade` ∈ {2, 3, 4}; `idh` and `1p19q` codes are 0 = wildtype / non-codeleted, 1 = mutant / codeleted; `ki_67` codes are 0 = ≤10 %, 1 = > 10 %.

## 4. Data availability

Individual patient MRI and clinical data used to train the deposited models are not distributed with this repository because of institutional review board restrictions and patient privacy regulations. De-identified data may be shared upon reasonable request as described in the manuscript's Resource Availability section.

## 5. License

MIT — see `LICENSE`.
