"""Generate aggregate subgroup statistics for Perfusion Habitat Ratios (PHRs).

Produces the ``summary_statistics.csv`` accompanying the manuscript. The output
mirrors Table 2 of the paper and contains no per-patient rows, only subgroup
aggregates (median, Q1, Q3, n) and the associated non-parametric P values.

Usage:
    python summary_statistics.py \\
        --input-csv path/to/label_with_quantization.csv \\
        --output-csv analysis/data/summary_statistics.csv

The input CSV is the patient-level table produced by ``compute_ratios.py``. It
must contain: ``patient_id, phase, who_grade, idh, _1p19q, ki_67,
dVAE_ratio_1 ... dVAE_ratio_8``.

The output CSV columns are:
    endpoint, subgroup, habitat, n, median_pct, q1_pct, q3_pct, p_value_within_endpoint
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


HABITATS = list(range(1, 9))
FEATURE_COL_FMT = "dVAE_ratio_{h}"


def _pct(series: pd.Series) -> np.ndarray:
    """Convert proportion to percentage."""
    return series.to_numpy(dtype=float) * 100.0


def _row(endpoint: str, subgroup: str, habitat: int, values: np.ndarray, p: float | None):
    q = np.percentile(values, [25, 50, 75])
    return {
        "endpoint": endpoint,
        "subgroup": subgroup,
        "habitat": habitat,
        "n": int(values.size),
        "median_pct": float(q[1]),
        "q1_pct": float(q[0]),
        "q3_pct": float(q[2]),
        "p_value_within_endpoint": (float(p) if p is not None else np.nan),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the aggregate summary table (216 rows)."""
    # Restrict to primary cohort (drop external cohort if any).
    df = df[df["phase"].isin({"train", "test"})].copy()

    # Ki-67 binarization at 10 %.
    df["ki67_bin"] = (df["ki_67"] > 10).astype(int)

    rows: list[dict] = []

    # --- WHO grade (Kruskal-Wallis, 3 groups) -----------------------------
    grade_levels = [(2, "Grade 2"), (3, "Grade 3"), (4, "Grade 4")]
    for h in HABITATS:
        col = FEATURE_COL_FMT.format(h=h)
        groups = [_pct(df.loc[df["who_grade"] == g, col]) for g, _ in grade_levels]
        _, p = kruskal(*groups)
        for (g, label), vals in zip(grade_levels, groups):
            rows.append(_row("WHO grade", label, h, vals, p))

    # --- IDH mutation (Mann-Whitney U) ------------------------------------
    idh_levels = [(0, "IDH-wildtype"), (1, "IDH-mutant")]
    for h in HABITATS:
        col = FEATURE_COL_FMT.format(h=h)
        a = _pct(df.loc[df["idh"] == 0, col])
        b = _pct(df.loc[df["idh"] == 1, col])
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        for (v, label), vals in zip(idh_levels, [a, b]):
            rows.append(_row("IDH mutation", label, h, vals, p))

    # --- 1p/19q codeletion (Mann-Whitney U) -------------------------------
    codel_levels = [(0, "Non-codeleted"), (1, "Codeleted")]
    for h in HABITATS:
        col = FEATURE_COL_FMT.format(h=h)
        a = _pct(df.loc[df["_1p19q"] == 0, col])
        b = _pct(df.loc[df["_1p19q"] == 1, col])
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        for (v, label), vals in zip(codel_levels, [a, b]):
            rows.append(_row("1p/19q codeletion", label, h, vals, p))

    # --- Ki-67 (Mann-Whitney U, low <=10 vs high >10) --------------------
    ki_levels = [(0, "Low (<=10%)"), (1, "High (>10%)")]
    for h in HABITATS:
        col = FEATURE_COL_FMT.format(h=h)
        a = _pct(df.loc[df["ki67_bin"] == 0, col])
        b = _pct(df.loc[df["ki67_bin"] == 1, col])
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        for (v, label), vals in zip(ki_levels, [a, b]):
            rows.append(_row("Ki-67", label, h, vals, p))

    out = pd.DataFrame(rows)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-csv", type=Path, required=True,
                   help="Patient-level CSV from compute_ratios.py")
    p.add_argument("--output-csv", type=Path,
                   default=Path(__file__).with_name("data") / "summary_statistics.csv",
                   help="Where to write the aggregate summary CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    summary = build_summary(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(summary)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
