"""Noise injection attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel

_SUPPORTED_KINDS = frozenset({"gaussian", "uniform"})


@dataclass
class NoiseAttack(Attack):
    """Mix random noise into the waveform at a target signal-to-noise ratio.

    Black-box attack: no gradient access needed, works on any ASR model. The
    output preserves the input shape and dtype, and is clipped to
    ``[-1, 1]`` to stay in valid amplitude range.

    Parameters
    ----------
    snr_db : float
        Target signal-to-noise ratio in dB. Lower = noisier. Common reference
        points: 20 dB = mild, 10 dB = noticeable but intelligible, 0 dB =
        noise as loud as the signal, negative = noise dominates.
    kind : str
        Noise distribution. ``"gaussian"`` produces i.i.d. samples from a
        normal distribution (white noise, the standard choice).
        ``"uniform"`` produces i.i.d. samples from a uniform distribution
        with matching variance.
    seed : int | None
        Optional RNG seed for reproducibility (useful in tests).
    """

    name: ClassVar[str] = "noise"
    snr_db: float = 20.0
    kind: str = "gaussian"
    seed: int | None = field(default=None)

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        if self.kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"Unknown noise kind {self.kind!r}; "
                f"expected one of {sorted(_SUPPORTED_KINDS)}."
            )

        x = np.asarray(audio, dtype=np.float32)
        rng = np.random.default_rng(self.seed)

        if self.kind == "gaussian":
            noise = rng.standard_normal(x.shape).astype(np.float32)
        else:  # uniform: range ±sqrt(3) has unit variance
            bound = float(np.sqrt(3.0))
            noise = rng.uniform(-bound, bound, size=x.shape).astype(np.float32)

        # Scale the noise to hit the target SNR.
        signal_power = float(np.mean(x**2))
        if signal_power <= 0.0:
            # Silent input: SNR is undefined, so we leave the signal alone.
            return x

        target_noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        current_noise_power = float(np.mean(noise**2))
        if current_noise_power > 0.0:
            noise = noise * float(np.sqrt(target_noise_power / current_noise_power))

        adv = x + noise.astype(np.float32)
        return np.clip(adv, -1.0, 1.0).astype(np.float32)
