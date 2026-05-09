"""Fast Gradient Sign Method attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class FGSMAttack(Attack):
    """Single-step Fast Gradient Sign Method on the loss w.r.t. the waveform."""

    name: ClassVar[str] = "fgsm"
    epsilon: float = 0.01

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("FGSMAttack.perturb is not implemented yet.")
