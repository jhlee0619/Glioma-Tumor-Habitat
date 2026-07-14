"""
Transforms for DSC voxel-wise curves (numpy -> torch for DataLoader).
"""
import numpy as np
import torch


class DSCVoxelTransform:
    """Single-voxel DSC curve: (T,) or (1, T) -> (1, T) float tensor."""

    def __call__(self, d):
        dsc = d["dsc_signal"]
        if not isinstance(dsc, torch.Tensor):
            dsc = torch.from_numpy(np.asarray(dsc, dtype=np.float32))
        if dsc.dim() == 1:
            dsc = dsc.unsqueeze(0)
        out = {"dsc_signal": dsc}
        if "dsc_signal_maxv" in d and d["dsc_signal_maxv"] is not None:
            m = d["dsc_signal_maxv"]
            if not isinstance(m, torch.Tensor):
                m = torch.from_numpy(np.asarray(m, dtype=np.float32))
            out["dsc_signal_maxv"] = m
        return out


def get_transforms_DSC():
    return DSCVoxelTransform()
