"""Fast Gradient Sign Method attack on ASR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class FGSMAttack(Attack):
    """Single-step Fast Gradient Sign Method on the loss w.r.t. the waveform.

    Untargeted by default: when ``target`` is ``None``, the model's clean
    transcription is used as the label and the perturbation is added in the
    direction that *increases* the CTC loss (pushes the model away from its
    own clean prediction). Pass an explicit ``target`` to run a targeted
    attack — the perturbation is then applied in the descent direction so
    the model's output drifts toward ``target``.

    The L_inf norm of the perturbation is bounded by ``epsilon``. The output
    is clamped to ``[clip_min, clip_max]`` (default ``[-1, 1]``) to keep the
    waveform in valid amplitude range.

    Restricted to models that expose a differentiable path from the waveform
    to the loss (see ``HFASRModel.supports_waveform_gradient``). In practice
    that means the wav2vec2 family of CTC models — wav2vec2, HuBERT, WavLM,
    UniSpeech, MMS, ... — which take ``input_values`` (raw waveform) directly.
    Models with mel-spectrogram inputs (Whisper, M-CTC-T, SpeechT5) need a
    torch-side feature extractor first, not wired up yet.
    """

    name: ClassVar[str] = "fgsm"
    epsilon: float = 0.01
    clip_min: float = -1.0
    clip_max: float = 1.0

    def perturb(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: HFASRModel | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        if model is None:
            raise ValueError("FGSMAttack requires a model (gradient-based attack).")
        if not model.supports_waveform_gradient:
            raise NotImplementedError(
                f"FGSMAttack requires a model that supports waveform-level "
                f"gradients; {model.model_id!r} (kind={model.kind!r}) does not. "
                "Currently supported: CTC models taking raw-waveform input "
                "(wav2vec2 family: wav2vec2, HuBERT, WavLM, UniSpeech, MMS, ...). "
                "M-CTC-T (mel-spec CTC) and seq2seq models (Whisper, SpeechT5) "
                "need a torch-side feature extractor."
            )

        audio_np = np.asarray(audio, dtype=np.float32)
        targeted = target is not None

        if not targeted:
            target = model.transcribe(audio_np, sample_rate=sample_rate).text
            if not target.strip():
                raise ValueError(
                    "FGSMAttack: clean transcription is empty; pass an "
                    "explicit `target` to run a targeted attack instead."
                )

        audio_tensor = torch.tensor(
            audio_np,
            dtype=torch.float32,
            device=model.device,
            requires_grad=True,
        )

        loss = model.loss(audio_tensor, target, sample_rate=sample_rate)
        loss.backward()

        if audio_tensor.grad is None:
            raise RuntimeError(
                "FGSMAttack: gradient w.r.t. audio is None. The model.loss path "
                "did not produce gradients — check model.supports_waveform_gradient."
            )

        # Targeted: step against the gradient (descent toward `target`).
        # Untargeted: step with the gradient (ascent away from clean prediction).
        direction = -1.0 if targeted else 1.0
        with torch.no_grad():
            adversarial = audio_tensor + direction * self.epsilon * audio_tensor.grad.sign()
            adversarial = adversarial.clamp(self.clip_min, self.clip_max)

        return adversarial.detach().cpu().numpy().astype(np.float32)
