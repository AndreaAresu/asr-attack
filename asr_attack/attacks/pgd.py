"""Projected Gradient Descent attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class PGDAttack(Attack):
    """Iterative Projected Gradient Descent in the L_inf ball of radius `epsilon`.

    At each of ``n_steps`` iterations:

    1. Recompute the gradient of the loss w.r.t. the current adversarial audio.
    2. Step in the direction of ``sign(grad)`` with step size ``alpha``
       (untargeted: ascent; targeted: descent toward ``target``).
    3. Project the result back into the L_inf ball of radius ``epsilon``
       around the original audio, then clamp to ``[clip_min, clip_max]``.

    Optional ``random_start`` initializes the adversarial audio with uniform
    noise in ``[-epsilon, +epsilon]`` around the original, before stepping.
    This makes PGD strictly stronger than FGSM in expectation: it explores
    multiple starting points and many small-step refinements within the same
    L_inf budget, rather than betting everything on a single linearization.

    Restricted, like FGSM, to models with ``supports_waveform_gradient``
    (wav2vec2 family + Whisper today).
    """

    name: ClassVar[str] = "pgd"
    epsilon: float = 0.01
    alpha: float = 0.001
    n_steps: int = 40
    random_start: bool = True
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed: int | None = field(default=None)

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        if model is None:
            raise ValueError("PGDAttack requires a model (gradient-based attack).")
        if not model.supports_waveform_gradient:
            raise NotImplementedError(
                f"PGDAttack requires a model that supports waveform-level "
                f"gradients; {model.model_id!r} (kind={model.kind!r}) does not. "
                "Currently supported: CTC models taking raw-waveform input "
                "(wav2vec2 family: wav2vec2, HuBERT, WavLM, UniSpeech, MMS, ...) "
                "and Whisper (via the torch-side mel extractor). "
                "M-CTC-T and non-Whisper seq2seq models need a torch-side "
                "feature extractor."
            )

        x_orig = np.asarray(audio, dtype=np.float32)
        targeted = target is not None

        # Pin the target once. In untargeted mode we use the model's clean
        # prediction; recomputing it each step would let the target drift
        # along with the perturbation, which is not what PGD attacks.
        if not targeted:
            target = model.transcribe(x_orig, sample_rate=sample_rate).text
            if not target.strip():
                raise ValueError(
                    "PGDAttack: clean transcription is empty; pass an explicit "
                    "`target` to run a targeted attack instead."
                )

        rng = np.random.default_rng(self.seed)
        if self.random_start:
            init_noise = rng.uniform(
                -self.epsilon, self.epsilon, size=x_orig.shape
            ).astype(np.float32)
            x_adv = x_orig + init_noise
        else:
            x_adv = x_orig.copy()
        x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        lo = np.maximum(x_orig - self.epsilon, self.clip_min).astype(np.float32)
        hi = np.minimum(x_orig + self.epsilon, self.clip_max).astype(np.float32)

        direction = -1.0 if targeted else 1.0

        for _ in range(self.n_steps):
            x_torch = torch.tensor(
                x_adv,
                dtype=torch.float32,
                device=model.device,
                requires_grad=True,
            )
            loss = model.loss(x_torch, target, sample_rate=sample_rate)
            loss.backward()
            if x_torch.grad is None:
                raise RuntimeError(
                    "PGDAttack: gradient w.r.t. audio is None. The model.loss "
                    "path did not produce gradients — check "
                    "model.supports_waveform_gradient."
                )
            grad_sign = x_torch.grad.sign().detach().cpu().numpy()

            x_adv = x_adv + direction * self.alpha * grad_sign
            # Projection: stay inside the L_inf ball around x_orig AND inside
            # the valid amplitude range.
            x_adv = np.clip(x_adv, lo, hi)

        return x_adv.astype(np.float32)
