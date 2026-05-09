"""Environmental degradation attack: convolutional reverb plus background noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class EnvironmentAttack(Attack):
    """Convolve with an impulse response and/or mix in a background recording."""

    name: ClassVar[str] = "environment"
    ir_path: str | None = None
    background_path: str | None = None
    snr_db: float = 15.0

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("EnvironmentAttack.perturb is not implemented yet.")
