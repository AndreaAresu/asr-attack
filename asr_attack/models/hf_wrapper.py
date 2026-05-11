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

    @property
    def supports_waveform_gradient(self) -> bool:
        """True if gradient-based attacks on the raw waveform are supported.

        Holds for CTC models whose forward takes ``input_values`` directly
        (the wav2vec2 family: wav2vec2, HuBERT, WavLM, UniSpeech, SEW,
        data2vec-audio, MMS, ...). Returns False for CTC models whose input
        is a mel-spectrogram (M-CTC-T) and for seq2seq models like Whisper —
        both would need a torch-side feature extractor to make the gradient
        flow back to the waveform.
        """
        if self._model is None:
            return False
        if self._kind != "ctc":
            return False
        main_input = getattr(self._model, "main_input_name", "")
        return main_input == "input_values"

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

        When ``audio`` is a ``torch.Tensor`` and the model is CTC-style, the
        whole pipeline runs in torch (resampling, normalization, forward) so
        the gradient flows back to the input waveform — this is the path
        exercised by white-box attacks like FGSM/PGD. Otherwise the standard
        processor handles the input numerically and the returned loss is a
        scalar without a usable gradient w.r.t. the waveform.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Use HFASRModel.from_pretrained(...).")

        src_sr = sample_rate if sample_rate is not None else self.sample_rate

        if isinstance(audio, torch.Tensor) and self.supports_waveform_gradient:
            return self._ctc_loss_torch(audio, target, src_sr)

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

    def _ctc_loss_torch(
        self,
        audio_tensor: torch.Tensor,
        target: str,
        src_sr: int,
    ) -> torch.Tensor:
        """CTC loss path that keeps the autograd chain attached to the waveform."""
        if not self.supports_waveform_gradient:
            raise NotImplementedError(
                f"Model {self.model_id!r} does not support waveform-level "
                f"gradients (kind={self._kind!r}, main_input_name="
                f"{getattr(self._model, 'main_input_name', '?')!r}). "
                "Currently only CTC models with raw-waveform input are wired "
                "up (wav2vec2 family: wav2vec2, HuBERT, WavLM, UniSpeech, "
                "MMS, ...). M-CTC-T (mel-spec input) and seq2seq models "
                "(Whisper, SpeechT5) need a torch-side feature extractor."
            )
        audio_tensor = audio_tensor.to(device=self.device, dtype=torch.float32)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if src_sr != self.sample_rate:
            # torchaudio's resample is conv1d under the hood — differentiable.
            audio_tensor = TAF.resample(
                audio_tensor, orig_freq=src_sr, new_freq=self.sample_rate
            )

        # Replicate the wav2vec2 feature extractor's per-utterance zero-mean,
        # unit-variance normalization in torch. Matches the numpy formula
        # `(x - x.mean()) / sqrt(x.var() + 1e-7)` exactly (population variance).
        feature_extractor = getattr(self._processor, "feature_extractor", None)
        if feature_extractor is not None and getattr(feature_extractor, "do_normalize", False):
            mean = audio_tensor.mean(dim=-1, keepdim=True)
            var = audio_tensor.var(dim=-1, keepdim=True, unbiased=False)
            audio_tensor = (audio_tensor - mean) / torch.sqrt(var + 1e-7)

        tokenizer = self._processor.tokenizer
        label_ids = tokenizer(target, return_tensors="pt").input_ids.to(self.device)

        outputs = self._model(input_values=audio_tensor, labels=label_ids)
        return outputs.loss
