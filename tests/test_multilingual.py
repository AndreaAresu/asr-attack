"""Tests for the unified ``language=`` kwarg on HFASRModel.

Threading a language through the wrapper exercises two very different
mechanisms under the hood: Whisper consumes the language as a runtime
prompt token, MMS as a per-language adapter + LM head loaded into the
model weights. Both are tested here.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch

datasets = pytest.importorskip("datasets")

from asr_attack.models.hf_wrapper import HFASRModel

DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"


@pytest.fixture(scope="module")
def librispeech_sample() -> dict:
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


# ---------------------------------------------------------------------------
# Fast: just the attribute plumbing
# ---------------------------------------------------------------------------


def test_language_argument_is_stored_without_load() -> None:
    """Constructor accepts language without loading anything."""
    model = HFASRModel(model_id="dummy", device="cpu", language="it")
    assert model.language == "it"


def test_language_default_is_none() -> None:
    model = HFASRModel(model_id="dummy", device="cpu")
    assert model.language is None


# ---------------------------------------------------------------------------
# Whisper: language is a runtime prompt token
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def whisper_default() -> HFASRModel:
    return HFASRModel.from_pretrained("openai/whisper-tiny", device="cpu")


@pytest.fixture(scope="module")
def whisper_en() -> HFASRModel:
    return HFASRModel.from_pretrained(
        "openai/whisper-tiny", device="cpu", language="en"
    )


@pytest.fixture(scope="module")
def whisper_it() -> HFASRModel:
    return HFASRModel.from_pretrained(
        "openai/whisper-tiny", device="cpu", language="it"
    )


@pytest.mark.slow
def test_whisper_default_and_explicit_en_match_on_english_audio(
    whisper_default: HFASRModel,
    whisper_en: HFASRModel,
    librispeech_sample: dict,
) -> None:
    """On clean English audio, whisper-tiny's auto-detect lands on English,
    so default and language='en' must produce the same transcription."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    text_default = whisper_default.transcribe(audio, sample_rate=sr).text
    text_en = whisper_en.transcribe(audio, sample_rate=sr).text
    assert text_default == text_en


@pytest.mark.slow
def test_whisper_language_it_forces_italian_on_english_audio(
    whisper_default: HFASRModel,
    whisper_it: HFASRModel,
    librispeech_sample: dict,
) -> None:
    """Forcing language='it' should produce a different (Italian-shaped)
    output than the default auto-detect — the prompt token is reaching
    Whisper's generate() and overriding language detection.
    """
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    text_default = whisper_default.transcribe(audio, sample_rate=sr).text
    text_it = whisper_it.transcribe(audio, sample_rate=sr).text
    assert text_default != text_it, (
        f"language='it' produced the same output as default — "
        f"prompt token was not propagated. default={text_default!r}, it={text_it!r}"
    )


# ---------------------------------------------------------------------------
# MMS: language picks an adapter + tokenizer vocab at load time
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mms_default() -> HFASRModel:
    return HFASRModel.from_pretrained("facebook/mms-1b-fl102", device="cpu")


@pytest.fixture(scope="module")
def mms_ita() -> HFASRModel:
    return HFASRModel.from_pretrained(
        "facebook/mms-1b-fl102", device="cpu", language="ita"
    )


@pytest.mark.slow
def test_mms_default_tokenizer_target_lang(mms_default: HFASRModel) -> None:
    """The fl102 checkpoint defaults to English; our wrapper preserves that."""
    assert mms_default._processor.tokenizer.target_lang == "eng"


@pytest.mark.slow
def test_mms_explicit_language_switches_tokenizer_and_adapter(
    mms_ita: HFASRModel,
) -> None:
    """Passing language='ita' must switch the tokenizer vocab and load the
    Italian adapter — verified through the tokenizer's `target_lang` attribute.
    """
    assert mms_ita._processor.tokenizer.target_lang == "ita"
    assert mms_ita.kind == "ctc"
    assert mms_ita.supports_waveform_gradient is True


@pytest.mark.slow
def test_mms_different_languages_produce_different_transcriptions(
    mms_default: HFASRModel,
    mms_ita: HFASRModel,
    librispeech_sample: dict,
) -> None:
    """English audio decoded with the eng adapter vs the ita adapter should
    produce different transcriptions — different LM heads, different vocabs."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    text_eng = mms_default.transcribe(audio, sample_rate=sr).text
    text_ita = mms_ita.transcribe(audio, sample_rate=sr).text
    assert text_eng != text_ita, (
        f"eng adapter and ita adapter gave the same output — "
        f"adapter switch did not take effect. eng={text_eng!r}, ita={text_ita!r}"
    )
