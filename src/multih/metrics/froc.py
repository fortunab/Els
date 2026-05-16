from __future__ import annotations

import numpy as np


def froc_score(sensitivities: np.ndarray, false_positives_per_image: np.ndarray, fp_points: tuple[float, ...] = (0.125, 0.25, 0.5, 1, 2, 4, 8)) -> float:
    """Approximate FROC score as mean interpolated sensitivity at standard FP/image points.

    For true lesion detection, pass arrays obtained from detection thresholds.
    For classification-only papers, report this only if lesion/localization outputs exist.
    """
    order = np.argsort(false_positives_per_image)
    fps = np.asarray(false_positives_per_image)[order]
    sens = np.asarray(sensitivities)[order]
    interp = np.interp(fp_points, fps, sens, left=sens[0], right=sens[-1])
    return float(np.mean(interp))
