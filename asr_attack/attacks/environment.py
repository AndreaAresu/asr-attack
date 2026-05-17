"""Environmental degradation attack: convolutional reverb plus background noise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as TAF

from asr_attack.attacks.base import Attack

if TYPE_CHECKING:
    from asr_attack.models.hf_wrapper import HFASRModel


@dataclass
class EnvironmentAttack(Attack):
    """Simulate "in-the-wild" recording conditions: room reverb + background noise.

    Black-box attack: no gradient access needed, works on any ASR model.

    The attack applies up to two transformations, in order:

    1. **Reverb** (if ``ir_path`` is set): convolve the input waveform with the
       impulse response (IR) loaded from ``ir_path``. The IR is normalized to
       unit energy and the convolved output is RMS-matched to the input, so
       the loudness is preserved (only the spectral / temporal character
       changes).
    2. **Background mixing** (if ``background_path`` is set): load the
       background recording, loop or truncate to match the input duration,
       and mix it in at the target ``snr_db``.

    At least one of ``ir_path`` and ``background_path`` must be provided.
    Both source files can have any sample rate — they are resampled to
    ``sample_rate`` with ``torchaudio.functional.resample``. Stereo files are
    downmixed to mono. The output is clipped to ``[-1, 1]``.
    """

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
        if self.ir_path is None and self.background_path is None:
            raise ValueError(
                "EnvironmentAttack needs at least one of ir_path / background_path."
            )

        x = np.asarray(audio, dtype=np.float32)

        if self.ir_path is not None:
            ir = _load_mono_wav(self.ir_path, target_sr=sample_rate)
            x = _apply_ir(x, ir)

        if self.background_path is not None:
            bg = _load_mono_wav(self.background_path, target_sr=sample_rate)
            x = _mix_background(x, bg, snr_db=self.snr_db)

        return np.clip(x, -1.0, 1.0).astype(np.float32)


def _load_mono_wav(path: str | Path, target_sr: int) -> np.ndarray:
    """Load a WAV file, downmix to mono, resample to ``target_sr``."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1).astype(np.float32)
    if int(sr) != int(target_sr):
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        audio = TAF.resample(tensor, orig_freq=int(sr), new_freq=int(target_sr)).numpy()
    return np.asarray(audio, dtype=np.float32)


def _apply_ir(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve ``signal`` with ``ir`` and match the result's RMS to the input.

    The IR is normalized to unit energy first so the magnitude of the IR file
    doesn't bleed into the output loudness. The convolution result is then
    truncated to the original signal length (we don't want to grow the audio
    by ``len(ir) - 1`` samples).
    """
    ir = ir / float(np.sqrt(np.sum(ir**2)) + 1e-9)
    sig_t = torch.from_numpy(np.ascontiguousarray(signal))
    ir_t = torch.from_numpy(np.ascontiguousarray(ir))
    convolved = TAF.fftconvolve(sig_t, ir_t, mode="full").numpy()[: signal.shape[-1]]

    in_rms = float(np.sqrt(np.mean(signal**2))) + 1e-9
    out_rms = float(np.sqrt(np.mean(convolved**2))) + 1e-9
    return (convolved * (in_rms / out_rms)).astype(np.float32)


def _mix_background(signal: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    """Loop/truncate ``background`` to match ``signal`` length, scale to ``snr_db``."""
    n = signal.shape[-1]
    if background.shape[-1] < n:
        reps = int(np.ceil(n / background.shape[-1]))
        background = np.tile(background, reps)
    background = background[:n]

    sig_p = float(np.mean(signal**2))
    if sig_p <= 0.0:
        return signal.astype(np.float32)
    target_bg_p = sig_p / (10.0 ** (snr_db / 10.0))
    bg_p = float(np.mean(background**2))
    if bg_p > 0.0:
        background = background * float(np.sqrt(target_bg_p / bg_p))
    return (signal + background).astype(np.float32)
