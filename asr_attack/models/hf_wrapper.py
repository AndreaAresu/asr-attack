"""Unified wrapper around Hugging Face ASR models (Whisper, wav2vec2, MMS, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchaudio.functional as TAF
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

# Architectures that use a CTC head over raw waveform input (`input_values`).
_CTC_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "wav2vec2",
        "wav2vec2-conformer",
        "hubert",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "sew",
        "sew-d",
        "data2vec-audio",
        "mctct",
        "mms",
    }
)

# Encoder-decoder architectures that consume `input_features` (mel specs).
_SEQ2SEQ_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "whisper",
        "speech-encoder-decoder",
        "speech_to_text",
        "speech_to_text_2",
    }
)


@dataclass
class TranscriptionResult:
    """Output of `HFASRModel.transcribe`."""

    text: str
    sample_rate: int = 16000
    logits: np.ndarray | None = None


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_float32_array(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        return audio.detach().to(torch.float32).cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    resampled = TAF.resample(tensor, orig_freq=src_sr, new_freq=dst_sr)
    return resampled.numpy()


class HFASRModel:
    """Uniform wrapper around an ASR model from the Hugging Face Hub.

    Hides the difference between CTC models (wav2vec2, MMS, ...) and seq2seq
    models (Whisper) so attacks and the benchmark can treat them identically.
    Loads the matching `AutoProcessor` and inspects the config to dispatch
    between `AutoModelForCTC` and `AutoModelForSpeechSeq2Seq`.
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "float32",
        sample_rate: int = 16000,
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
        self.dtype = dtype
        self.sample_rate = sample_rate
        self._model: Any = None
        self._processor: Any = None
        self._kind: str = ""

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> HFASRModel:
        """Load processor and model weights from the Hugging Face Hub."""
        instance = cls(model_id=model_id, **kwargs)
        instance._load()
        return instance

    @property
    def kind(self) -> str:
        """Either ``"ctc"`` or ``"seq2seq"`` once the model is loaded."""
        return self._kind

    def _padding_kwargs(self) -> dict[str, Any]:
        # Whisper-style encoders need a fixed 30s mel-spec tensor; CTC encoders
        # work with the longest waveform in the batch.
        if self._kind == "seq2seq":
            return {"padding": "max_length", "truncation": True}
        return {"padding": True}

    def _load(self) -> None:
        torch_dtype = getattr(torch, self.dtype)
        config = AutoConfig.from_pretrained(self.model_id)
        model_type = (getattr(config, "model_type", "") or "").lower()

        if model_type in _SEQ2SEQ_MODEL_TYPES:
            self._kind = "seq2seq"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=torch_dtype
            )
        elif model_type in _CTC_MODEL_TYPES:
            self._kind = "ctc"
            model = AutoModelForCTC.from_pretrained(self.model_id, torch_dtype=torch_dtype)
        else:
            # Unknown family: try seq2seq first, then fall back to CTC.
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id, torch_dtype=torch_dtype
                )
                self._kind = "seq2seq"
            except (ValueError, OSError, KeyError):
                model = AutoModelForCTC.from_pretrained(
                    self.model_id, torch_dtype=torch_dtype
                )
                self._kind = "ctc"

        self._model = model.to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        feature_extractor = getattr(self._processor, "feature_extractor", None)
        fe_sr = getattr(feature_extractor, "sampling_rate", None)
        if fe_sr:
            self.sample_rate = int(fe_sr)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> TranscriptionResult:
        """Run inference on a single waveform."""
        return self.transcribe_batch([audio], sample_rate=sample_rate)[0]

    def transcribe_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int | None = None,
    ) -> list[TranscriptionResult]:
        """Run inference on a batch of waveforms."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Use HFASRModel.from_pretrained(...).")

        src_sr = sample_rate if sample_rate is not None else self.sample_rate
        prepared = [_to_float32_array(a) for a in audios]
        if src_sr != self.sample_rate:
            prepared = [_resample(a, src_sr, self.sample_rate) for a in prepared]

        inputs = self._processor(
            prepared,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            **self._padding_kwargs(),
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if self._kind == "seq2seq":
                generated = self._model.generate(**inputs)
                texts = self._processor.batch_decode(generated, skip_special_tokens=True)
                return [
                    TranscriptionResult(text=t.strip(), sample_rate=self.sample_rate)
                    for t in texts
                ]

            logits = self._model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            texts = self._processor.batch_decode(pred_ids)
            logits_np = logits.detach().to(torch.float32).cpu().numpy()
            return [
                TranscriptionResult(
                    text=t.strip(),
                    sample_rate=self.sample_rate,
                    logits=logits_np[i],
                )
                for i, t in enumerate(texts)
            ]

    def loss(
        self,
        audio: np.ndarray | torch.Tensor,
        target: str,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Teacher-forcing loss between the model's output on `audio` and `target`.

        For CTC models the gradient flows back through `input_values` to the
        waveform, supporting white-box attacks. For seq2seq models the standard
        feature extractor is non-differentiable, so the returned scalar is
        useful for diagnostics but attacks need a torch-side mel computation.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Use HFASRModel.from_pretrained(...).")

        src_sr = sample_rate if sample_rate is not None else self.sample_rate
        audio_np = _to_float32_array(audio)
        if src_sr != self.sample_rate:
            audio_np = _resample(audio_np, src_sr, self.sample_rate)

        inputs = self._processor(
            audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            **self._padding_kwargs(),
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        label_ids = tokenizer(target, return_tensors="pt").input_ids.to(self.device)

        outputs = self._model(**inputs, labels=label_ids)
        return outputs.loss
