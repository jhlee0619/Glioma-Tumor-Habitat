# Perfusion Habitat Analysis for Diffuse Glioma — Anonymous Code Repository

Accompanies the Cell Reports Medicine manuscript **"Perfusion-based Habitat
Analysis for Visualization and Quantification of Hemodynamic Heterogeneity
in Diffuse Glioma."** The identifiers of the authors and their institution
have been removed from this repository (see §5 below).

The materials here let a reviewer

1. reload the trained discrete variational autoencoder (dVAE) and the four
   Random Forest classifiers,
2. reproduce Tables 2 and 3 from a user-supplied patient-level CSV, and
3. inspect the aggregate subgroup statistics that underlie Table 2 and the
   habitat curves shown in Figures 1A–D.

## 1. Layout

```
├── README.md              this file
├── LICENSE                MIT
├── requirements.txt       pip dependencies
├── model/
│   ├── configs/vqgan_sep.yaml       architecture used to train the checkpoint
│   ├── taming/                       dVAE source (encoder/decoder, VQ, dataset)
│   ├── encode.py                     whole-cohort inference
│   ├── project_utils.py              instantiate_from_config helper
│   └── weights/
│       ├── dvae_checkpoint.ckpt      17 MB, epoch 74
│       └── rf/                       4 Random Forest classifiers
└── analysis/
    ├── train_models.py               RF training + evaluation + DeLong test
    ├── compute_ratios.py             PHR extraction from habitat NIfTI
    ├── summary_statistics.py         generates the aggregate CSV
    └── data/
        ├── summary_statistics.csv    Table 2 (72 rows, no per-patient data)
        ├── habitat_curves/dVAE_quantiles.pkl        Figure 1C, D source
        └── reference_tissue/*.pkl                    Figure 1A, B source
```

## 2. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reload the deposited dVAE and run a forward pass.
python -c "
import sys; sys.path.insert(0, 'model')
import torch
from omegaconf import OmegaConf
from project_utils import instantiate_from_config
cfg = OmegaConf.load('model/configs/vqgan_sep.yaml')
cfg.model.params.sane_index_shape = True
m = instantiate_from_config(cfg.model)
sd = torch.load('model/weights/dvae_checkpoint.ckpt', map_location='cpu',
                weights_only=False)['state_dict']
m.load_state_dict(sd, strict=False)
m.eval()
with torch.no_grad():
    _, _, info = m.encode(torch.rand(4, 1, 60))
print('habitat codes:', info[2].tolist())
"

# Reproduce Table 3 (needs the patient-level CSV, IRB-restricted).
cd analysis
python train_models.py --input-csv <label_with_quantization.csv> \
       --model-dir ./trained_rf --output-dir ./reports
```

## 3. Reproducing tables and figures

### Table 1 — Patient characteristics

Aggregates demographic and pathologic columns of the source clinical CSV
(`patient_id, age, sex, who_grade, idh, _1p19q, ki_67, kps, eor, mgmt`). It
does not depend on any model output — reproduce with pandas
`groupby("phase").describe()`.

### Table 2 — PHR distribution by subgroup

Precomputed at `analysis/data/summary_statistics.csv` (72 rows, one per
(endpoint × subgroup × habitat) with `median_pct, q1_pct, q3_pct, n`, and
the P value from the corresponding non-parametric test — Kruskal-Wallis for
WHO grade, Mann-Whitney U for the binary endpoints).

Regeneration:

```bash
cd analysis
python summary_statistics.py --input-csv <label_with_quantization.csv> \
       --output-csv data/summary_statistics.csv
```

Cross-check: Habitat 1 medians for Grade 2 / 3 / 4 should read
`0.15 / 0.44 / 2.21`, all with `P < .001`.

### Table 3 — Discriminatory performance

```bash
cd analysis
python train_models.py --input-csv <label_with_quantization.csv> \
       --model-dir ./trained_rf --output-dir ./reports
```

Outputs per-endpoint AUC (5000-iteration bootstrap 95% CIs), DeLong test P
values, and per-method sub-tables in `reports/model_<method>/`. The DPH row
should reproduce the paper values (AUC 0.90 IDH, 0.85 1p/19q, 0.81 WHO,
0.78 Ki-67). The RF classifiers at `model/weights/rf/*.joblib` were
produced by the same command with `random_state=42`; loading them yields
byte-identical `predict_proba` outputs across runs.

### Figure 1A, B — Reference tissue curves

Precomputed summaries at `analysis/data/reference_tissue/*.pkl` (one pickle
per tissue with keys `median_curve, q1_curve, q3_curve, peak_height,
percentage_recovery, n_voxels`). Plot inline:

```python
import pickle, matplotlib.pyplot as plt
for tissue in ('artery', 'gm', 'wm', 'csf', 'cp'):
    d = pickle.load(open(f'analysis/data/reference_tissue/{tissue}_curves.pkl', 'rb'))
    plt.plot(d['median_curve'], label=tissue)
    plt.fill_between(range(60), d['q1_curve'], d['q3_curve'], alpha=0.2)
plt.legend(); plt.savefig('fig1AB.png')
```

### Figure 1C, D — Habitat curves and bar plots

Habitat quantile curves at `analysis/data/habitat_curves/dVAE_quantiles.pkl`
(dict keyed by habitat `1..8`; each entry is `[q1_curve, median_curve,
q3_curve]`, each a 60-point array). Same three-line plotting pattern.

### Figure 2A — PHR bar plots stratified by subgroup

Reads `summary_statistics.csv` (medians and IQRs).

### Figure 2B — SHAP summary plots

```python
import joblib, shap, pandas as pd, matplotlib.pyplot as plt
rf = joblib.load('model/weights/rf/dVAE_who_grade_random_forest.joblib')
df = pd.read_csv('<label_with_quantization.csv>')
X = df[[f'dVAE_ratio_{i}' for i in range(1, 9)]].values
explainer = shap.TreeExplainer(rf)
shap.summary_plot(explainer.shap_values(X), X,
                  feature_names=[f'H{i}' for i in range(1, 9)], show=False)
plt.savefig('fig2B_who.png')
```

### Reproducing habitat NIfTI maps from raw DSC volumes

```bash
cd model
python encode.py --dataroot <path/to/patient/root> \
       --resume weights/dvae_checkpoint.ckpt -c configs/vqgan_sep.yaml
```

writes `dsc_clusters/dVAE_quantization.nii.gz` per patient. Feeding these
back into `analysis/compute_ratios.py` regenerates the patient-level CSV
used above.

## 4. Data availability

Individual patient MRI and clinical data are not distributed with this
repository because of institutional review board restrictions. Aggregate
subgroup statistics that reproduce Table 2 are provided at
`analysis/data/summary_statistics.csv`. See the manuscript's Resource
Availability section for the full statement.

## 5. Anonymization

Author, institutional, contact, IRB, funding, and patient identifiers have
been removed from every file. Reviewers can verify with:

```bash
grep -rEi "Junhyeok|Minseo|Ho Kang|Kyu Sung|K\.S\.C\.|kyuchoi|ent1127" .
grep -rEi "SNUH|Seoul National|Bundang|Healthcare AI Research" .
grep -rE "2212-077-1385|RS-2023-|RS-2024-|04-2024-|04-2025-" .
grep -rE "\+82-10-5042-7247|\+82-2-2072-1161" .
grep -rE "/data/jhlee|/home/jhlee" .
grep -rE "14472109|55726573|19745073" .
```

All commands return no matches (the identifiers listed in this README §5
itself are documentation of what was removed; they are the only exception
and are intentional).

Redaction categories: four author names (co-first authors and
corresponding author); five participating-institution names; corresponding
author's two email addresses, two phone numbers, credentials, and postal
address; IRB approval number; four funding grant identifiers; absolute
filesystem paths and Linux username; three hard-coded patient identifiers;
`[GitHub]` URL placeholder in the Code Availability paragraph; decompyle3
header comments in restored bytecode files.

The manuscript, cover letter, and Editorial Manager supplementary documents
are submitted to Cell Reports Medicine with real author identifiers
(single-blind review); only this code repository, hosted at
anonymous.4open.science, is anonymized.

## 6. License

MIT — see `LICENSE`.
