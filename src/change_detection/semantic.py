"""
Semantic change detection (V2 extension).

Compares two class masks (same grid) and reports per-class
appearance / disappearance / expansion / contraction.

Does NOT claim causality (e.g. never "new building constructed").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SemanticChangeResult:
    valid: bool
    message: str
    class_names: List[str]
    persistence_pct: Dict[str, float] = field(default_factory=dict)
    appearance_pct: Dict[str, float] = field(default_factory=dict)
    disappearance_pct: Dict[str, float] = field(default_factory=dict)
    expansion_pct: Dict[str, float] = field(default_factory=dict)
    contraction_pct: Dict[str, float] = field(default_factory=dict)
    summary: List[str] = field(default_factory=list)


def semantic_change(
    mask_t0: np.ndarray,
    mask_t1: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> SemanticChangeResult:
    if mask_t0 is None or mask_t1 is None:
        return SemanticChangeResult(
            valid=False, message="One or both masks missing.", class_names=class_names or []
        )
    a = np.asarray(mask_t0)
    b = np.asarray(mask_t1)
    if a.shape != b.shape:
        return SemanticChangeResult(
            valid=False,
            message=f"Semantic change unavailable: mask shape mismatch {a.shape} vs {b.shape}.",
            class_names=class_names or [],
        )
    if class_names is None:
        n = int(max(a.max(), b.max()) + 1)
        class_names = [f"Class_{i}" for i in range(n)]
    n = len(class_names)
    total = a.size
    persistence, appearance, disappearance, expansion, contraction = {}, {}, {}, {}, {}
    summary = []
    for c, name in enumerate(class_names):
        t0 = a == c
        t1 = b == c
        pers = np.logical_and(t0, t1).sum()
        app = np.logical_and(~t0, t1).sum()
        dis = np.logical_and(t0, ~t1).sum()
        persistence[name] = 100.0 * pers / total
        appearance[name] = 100.0 * app / total
        disappearance[name] = 100.0 * dis / total
        t0_n = max(int(t0.sum()), 1)
        expansion[name] = 100.0 * app / t0_n
        contraction[name] = 100.0 * dis / t0_n
        if appearance[name] > 1.0:
            summary.append(
                f"Semantic expansion detected in {name.lower()}-class pixels "
                f"(+{appearance[name]:.1f}% of scene)."
            )
        if disappearance[name] > 1.0:
            summary.append(
                f"Semantic contraction detected in {name.lower()}-class pixels "
                f"(-{disappearance[name]:.1f}% of scene)."
            )
    return SemanticChangeResult(
        valid=True,
        message="Semantic change comparison completed.",
        class_names=class_names,
        persistence_pct=persistence,
        appearance_pct=appearance,
        disappearance_pct=disappearance,
        expansion_pct=expansion,
        contraction_pct=contraction,
        summary=summary or ["No major semantic class shifts above 1% of scene."],
    )
