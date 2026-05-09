"""Projected Gradient Descent attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class PGDAttack(Attack):
    """Iterative Projected Gradient Descent in the L_inf ball of radius `epsilon`."""

    name: ClassVar[str] = "pgd"
    epsilon: float = 0.01
    alpha: float = 0.001
    n_steps: int = 40
    random_start: bool = True

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("PGDAttack.perturb is not implemented yet.")
