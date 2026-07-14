"""End-to-end inference for a single patient.

Given the voxel-wise habitat map produced by ``model/encode.py`` and a tumor
mask NIfTI, this script

  1. computes the 8-element Perfusion Habitat Ratio (PHR) vector, and
  2. predicts probabilities for WHO grade, IDH mutation, 1p/19q codeletion,
     and Ki-67 index using the deposited Random Forest classifiers.

Usage:
    python predict.py --habitat dVAE_quantization.nii.gz --mask tumor_mask.nii.gz

The Random Forest joblib files are expected at ``../model/weights/rf/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np

RF_DIR = Path(__file__).resolve().parent.parent / "model" / "weights" / "rf"
TARGETS = {
    "who_grade": "dVAE_who_grade_random_forest.joblib",
    "idh": "dVAE_idh_random_forest.joblib",
    "1p19q": "dVAE__1p19q_random_forest.joblib",
    "ki_67": "dVAE_ki_67_random_forest.joblib",
}
N_HABITATS = 8


def compute_phr(habitat_nii: str, mask_nii: str) -> np.ndarray:
    """Return the 8-element Perfusion Habitat Ratio for one patient."""
    habitat = nib.load(habitat_nii).get_fdata().astype(int)
    mask = nib.load(mask_nii).get_fdata() > 0
    voxels = habitat[mask]
    total = int(voxels.size)
    if total == 0:
        raise ValueError(f"Tumor mask '{mask_nii}' contains no positive voxels.")
    return np.array([(voxels == h).sum() / total for h in range(1, N_HABITATS + 1)])


def predict(phr: np.ndarray) -> dict[str, dict]:
    """Return per-endpoint class probabilities for a single PHR vector."""
    x = np.asarray(phr, dtype=float).reshape(1, -1)
    if x.shape != (1, N_HABITATS):
        raise ValueError(f"PHR vector must have length {N_HABITATS}, got {x.shape}.")
    out: dict[str, dict] = {}
    for name, fn in TARGETS.items():
        rf = joblib.load(RF_DIR / fn)
        probs = rf.predict_proba(x)[0]
        out[name] = {str(cls): float(p) for cls, p in zip(rf.classes_, probs)}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--habitat", required=True, help="Habitat NIfTI from model/encode.py")
    p.add_argument("--mask", required=True, help="Binary tumor mask NIfTI")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    phr = compute_phr(args.habitat, args.mask)
    print("Perfusion Habitat Ratio (H1..H8):")
    for i, r in enumerate(phr, 1):
        print(f"  H{i}: {r:.4f}")

    preds = predict(phr)
    print("\nClassification probabilities:")
    for endpoint, probs in preds.items():
        summary = ", ".join(f"P({cls})={p:.3f}" for cls, p in probs.items())
        print(f"  {endpoint}: {summary}")


if __name__ == "__main__":
    main()
