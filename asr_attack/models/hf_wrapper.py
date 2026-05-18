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
    """CUDA if available, else CPU. MPS is intentionally skipped as a default.

    Apple's MPS backend does not implement ``aten::_ctc_loss`` (needed by
    FGSM/PGD on every wav2vec2-family CTC model) and has had patchy support
    for ``torch.stft`` / FFT backward (the Whisper torch-mel path). Both
    would crash white-box attacks on Mac when the user didn't explicitly
    opt in. Inference-only and black-box workflows can still use MPS by
    passing ``device="mps"`` to ``HFASRModel.from_pretrained`` explicitly.
    """
    if torch.cuda.is_available():
        return "cuda"
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

        - CTC + ``input_values`` (the wav2vec2 family: wav2vec2, HuBERT,
          WavLM, UniSpeech, SEW, data2vec-audio, MMS, ...) → True.
        - Whisper (seq2seq) → True via a torch-side log-mel extractor.
        - Other seq2seq (SpeechT5, S2T) and M-CTC-T (CTC over mel) → False
          until their feature extractors are reimplemented in torch.
        """
        if self._model is None:
            return False
        if self._kind == "ctc":
            main_input = getattr(self._model, "main_input_name", "")
            return main_input == "input_values"
        if self._kind == "seq2seq":
            model_type = getattr(self._model.config, "model_type", "")
            return model_type == "whisper"
        return False

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
            if self._kind == "ctc":
                return self._ctc_loss_torch(audio, target, src_sr)
            if self._kind == "seq2seq":
                return self._whisper_loss_torch(audio, target, src_sr)

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

    def _whisper_log_mel_torch(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Compute Whisper's log-mel spectrogram in torch (autograd preserved).

        Input shape: ``[batch, samples]`` at ``self.sample_rate`` (16 kHz).
        Output shape: ``[batch, n_mels, n_frames]`` (typically ``[B, 80, 3000]``).

        Replicates the openai/whisper reference and HF's numpy version:
        pad/truncate to 30 s, periodic Hann window, ``torch.stft`` with
        ``center=True`` (reflect pad), drop the trailing frame to land on
        ``n_frames`` exactly, apply HF's precomputed Slaney mel filterbank,
        ``log10`` with a 1e-10 floor, clamp to peak-minus-8 dB, then rescale.
        """
        fe = self._processor.feature_extractor
        n_fft = int(fe.n_fft)
        hop_length = int(fe.hop_length)
        n_samples = int(fe.chunk_length * fe.sampling_rate)

        cur_len = audio_tensor.shape[-1]
        if cur_len < n_samples:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, n_samples - cur_len))
        elif cur_len > n_samples:
            audio_tensor = audio_tensor[..., :n_samples]

        window = torch.hann_window(n_fft, device=audio_tensor.device, dtype=audio_tensor.dtype)
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2  # [B, n_freq, n_frames]

        mel_filters_np = np.asarray(fe.mel_filters, dtype=np.float32)
        mel_filters = (
            torch.from_numpy(mel_filters_np)
            .to(device=audio_tensor.device, dtype=audio_tensor.dtype)
            .T  # HF stores (n_freq, n_mels); matmul wants (n_mels, n_freq).
        )
        mel_spec = mel_filters @ magnitudes  # [B, n_mels, n_frames]

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _whisper_loss_torch(
        self,
        audio_tensor: torch.Tensor,
        target: str,
        src_sr: int,
    ) -> torch.Tensor:
        """Whisper cross-entropy loss with autograd preserved from the waveform.

        Resamples to 16 kHz with ``torchaudio.functional.resample`` (conv1d,
        differentiable), computes the log-mel in torch, and forwards through
        the model with ``labels=`` so the loss is computed internally.

        Builds the labels in Whisper's training format:
        ``[<|en|>, <|transcribe|>, <|notimestamps|>, ...target..., <|endoftext|>]``.
        We deliberately omit the leading ``<|startoftranscript|>``: the model
        prepends ``decoder_start_token_id`` internally via ``shift_right`` when
        computing the loss. Language is currently fixed to English; the model
        and dataset we test against are English, and threading a ``language``
        kwarg through the attack API is out of scope for this change.
        """
        audio_tensor = audio_tensor.to(device=self.device, dtype=torch.float32)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if src_sr != self.sample_rate:
            audio_tensor = TAF.resample(
                audio_tensor, orig_freq=src_sr, new_freq=self.sample_rate
            )

        log_mel = self._whisper_log_mel_torch(audio_tensor)

        tokenizer = self._processor.tokenizer
        prompt_pairs = self._processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
        prefix_ids = [tok for _, tok in prompt_pairs]
        text_ids = tokenizer(target, add_special_tokens=False).input_ids
        eot_id = tokenizer.eos_token_id
        label_ids = torch.tensor(
            [prefix_ids + list(text_ids) + [eot_id]],
            device=self.device,
            dtype=torch.long,
        )

        outputs = self._model(input_features=log_mel, labels=label_ids)
        return outputs.loss
