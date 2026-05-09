"""Noise injection attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class NoiseAttack(Attack):
    """Mix random noise into the waveform at a target signal-to-noise ratio."""

    name: ClassVar[str] = "noise"
    snr_db: float = 20.0
    kind: str = "gaussian"

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("NoiseAttack.perturb is not implemented yet.")
