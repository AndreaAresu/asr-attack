"""End-to-end smoke test: load Whisper-tiny, transcribe a clean LibriSpeech sample, score WER.

Marked `slow` because it downloads ~75MB of model weights and a small dataset
on first run. Run with: ``uv run pytest -m slow``. Skip with: ``-m 'not slow'``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")

from asr_attack.metrics.wer import compute_error_rates, compute_wer
from asr_attack.models.hf_wrapper import HFASRModel, TranscriptionResult

MODEL_ID = "openai/whisper-tiny"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"


@pytest.fixture(scope="module")
def whisper_tiny() -> HFASRModel:
    return HFASRModel.from_pretrained(MODEL_ID, device="cpu")


@pytest.fixture(scope="module")
def librispeech_sample() -> dict:
    # `decode=False` returns the raw FLAC bytes so we can use soundfile and
    # avoid pulling in torchcodec just to decode one audio file in CI.
    audio_feature = datasets.Audio(decode=False)
    ds = datasets.load_dataset(DATASET_ID, "clean", split="validation").cast_column(
        "audio", audio_feature
    )
    row = ds[0]
    array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    return {
        "array": np.asarray(array, dtype=np.float32),
        "sampling_rate": int(sr),
        "text": row["text"],
    }


@pytest.mark.slow
def test_whisper_tiny_loads_as_seq2seq(whisper_tiny: HFASRModel) -> None:
    assert whisper_tiny.kind == "seq2seq"
    assert whisper_tiny.sample_rate == 16000


@pytest.mark.slow
def test_whisper_tiny_transcribes_clean_audio(
    whisper_tiny: HFASRModel,
    librispeech_sample: dict,
) -> None:
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    result = whisper_tiny.transcribe(audio, sample_rate=sr)

    assert isinstance(result, TranscriptionResult)
    assert isinstance(result.text, str)
    assert result.text.strip() != ""

    rates = compute_error_rates([reference], [result.text])
    # Whisper-tiny on clean LibriSpeech routinely lands well below 0.5.
    assert 0.0 <= rates.wer < 0.5, f"unexpectedly high WER: {rates.wer:.3f}"
    assert rates.n_samples == 1


@pytest.mark.slow
def test_whisper_tiny_batch_is_consistent(
    whisper_tiny: HFASRModel,
    librispeech_sample: dict,
) -> None:
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]

    results = whisper_tiny.transcribe_batch([audio, audio], sample_rate=sr)
    assert len(results) == 2
    assert results[0].text == results[1].text
    # Same waveform, single transcribe call agrees with the batched call.
    single = whisper_tiny.transcribe(audio, sample_rate=sr)
    assert single.text == results[0].text


@pytest.mark.slow
def test_whisper_tiny_resamples_when_input_sr_differs(
    whisper_tiny: HFASRModel,
    librispeech_sample: dict,
) -> None:
    audio = np.asarray(librispeech_sample["array"], dtype=np.float32)
    reference = librispeech_sample["text"]

    # Pretend the signal is 8 kHz (every other sample): wrapper must resample
    # back to 16 kHz before feeding the processor.
    downsampled = audio[::2]
    result = whisper_tiny.transcribe(downsampled, sample_rate=8000)

    assert result.text.strip() != ""
    wer = compute_wer([reference], [result.text])
    # Resampling from real 16 kHz down to 8 kHz then back up is lossy, but
    # the transcript should still be closer to the reference than random.
    assert wer < 0.9
