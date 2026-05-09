"""Abstract base class and factories for adversarial attacks on ASR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from asr_attack.attacks.environment import EnvironmentAttack
    from asr_attack.attacks.fgsm import FGSMAttack
    from asr_attack.attacks.noise import NoiseAttack
    from asr_attack.attacks.pgd import PGDAttack
    from asr_attack.models.hf_wrapper import HFASRModel


class Attack(ABC):
    """Abstract base class for adversarial attacks against ASR models.

    Concrete subclasses implement `perturb` to produce an adversarial waveform
    from a clean one. Use the classmethod factories (`Attack.fgsm`,
    `Attack.pgd`, `Attack.noise`, `Attack.environment`) to construct the
    standard attacks shipped with the library.
    """

    name: ClassVar[str] = "attack"

    @abstractmethod
    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        """Return an adversarial version of `audio`.

        Args:
            audio: 1-D float waveform in the range [-1, 1].
            sample_rate: sample rate of `audio` in Hz.
            model: optional surrogate ASR model used for gradient-based attacks.
            target: optional target transcript for targeted attacks.

        Returns:
            Adversarial waveform with the same shape and dtype as `audio`.
        """

    @classmethod
    def fgsm(cls, epsilon: float = 0.01) -> FGSMAttack:
        from asr_attack.attacks.fgsm import FGSMAttack

        return FGSMAttack(epsilon=epsilon)

    @classmethod
    def pgd(
        cls,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        n_steps: int = 40,
        random_start: bool = True,
    ) -> PGDAttack:
        from asr_attack.attacks.pgd import PGDAttack

        return PGDAttack(
            epsilon=epsilon,
            alpha=alpha,
            n_steps=n_steps,
            random_start=random_start,
        )

    @classmethod
    def noise(cls, snr_db: float = 20.0, kind: str = "gaussian") -> NoiseAttack:
        from asr_attack.attacks.noise import NoiseAttack

        return NoiseAttack(snr_db=snr_db, kind=kind)

    @classmethod
    def environment(
        cls,
        ir_path: str | None = None,
        background_path: str | None = None,
        snr_db: float = 15.0,
    ) -> EnvironmentAttack:
        from asr_attack.attacks.environment import EnvironmentAttack

        return EnvironmentAttack(
            ir_path=ir_path,
            background_path=background_path,
            snr_db=snr_db,
        )
