"""End-to-end FGSM test: attack wav2vec2-base on a clean LibriSpeech sample
and verify the WER goes up.

Marked `slow` because it downloads ~360MB of wav2vec2 weights and a small
LibriSpeech sample. Run with: ``uv run pytest -m slow``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")

from asr_attack import Attack
from asr_attack.attacks.fgsm import FGSMAttack
from asr_attack.metrics.wer import compute_wer
from asr_attack.models.hf_wrapper import HFASRModel

MODEL_ID = "facebook/wav2vec2-base-960h"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"


@pytest.fixture(scope="module")
def wav2vec2() -> HFASRModel:
    return HFASRModel.from_pretrained(MODEL_ID, device="cpu")


@pytest.fixture(scope="module")
def whisper_tiny() -> HFASRModel:
    return HFASRModel.from_pretrained("openai/whisper-tiny", device="cpu")


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


@pytest.mark.slow
def test_wav2vec2_base_loads_as_ctc(wav2vec2: HFASRModel) -> None:
    assert wav2vec2.kind == "ctc"
    assert wav2vec2.supports_waveform_gradient is True


@pytest.mark.slow
def test_fgsm_increases_wer(
    wav2vec2: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    clean_hyp = wav2vec2.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = Attack.fgsm(epsilon=0.02)
    adv_audio = attack.perturb(audio, sample_rate=sr, model=wav2vec2)

    # Shape and dtype preserved.
    assert adv_audio.shape == audio.shape
    assert adv_audio.dtype == np.float32

    # L_inf perturbation bounded by epsilon (with a tiny float tolerance).
    linf = float(np.abs(adv_audio - audio).max())
    assert linf <= attack.epsilon + 1e-6, f"L_inf {linf} exceeds epsilon {attack.epsilon}"

    adv_hyp = wav2vec2.transcribe(adv_audio, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    # Surfaced via -s for inspection.
    print(f"\n  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")
    print(f"  L_inf:     {linf:.4f}")

    assert adv_wer > clean_wer, (
        f"FGSM did not degrade transcription: clean WER {clean_wer:.3f}, "
        f"adv WER {adv_wer:.3f}"
    )


@pytest.mark.slow
def test_fgsm_attack_object_attributes() -> None:
    attack = Attack.fgsm(epsilon=0.05)
    assert isinstance(attack, FGSMAttack)
    assert attack.epsilon == 0.05
    assert attack.name == "fgsm"


def test_fgsm_rejects_models_without_waveform_gradient() -> None:
    """FGSM rejects any model whose `supports_waveform_gradient` is False
    (e.g. M-CTC-T, SpeechT5, S2T) with a clear NotImplementedError.

    We don't load a real unsupported model — the property is the contract,
    and an unloaded ``HFASRModel`` already exercises the False branch.
    """
    fake = HFASRModel(model_id="not-a-real-model", device="cpu")
    assert fake.supports_waveform_gradient is False

    attack = Attack.fgsm(epsilon=0.01)
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(NotImplementedError, match="waveform"):
        attack.perturb(audio, sample_rate=16000, model=fake)


@pytest.mark.slow
def test_whisper_tiny_supports_waveform_gradient(whisper_tiny: HFASRModel) -> None:
    assert whisper_tiny.kind == "seq2seq"
    assert whisper_tiny.supports_waveform_gradient is True


@pytest.mark.slow
def test_fgsm_increases_wer_on_whisper(
    whisper_tiny: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """FGSM via the torch-side mel extractor degrades Whisper-tiny's WER."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    clean_hyp = whisper_tiny.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = Attack.fgsm(epsilon=0.05)
    adv_audio = attack.perturb(audio, sample_rate=sr, model=whisper_tiny)

    assert adv_audio.shape == audio.shape
    assert adv_audio.dtype == np.float32
    linf = float(np.abs(adv_audio - audio).max())
    assert linf <= attack.epsilon + 1e-6

    adv_hyp = whisper_tiny.transcribe(adv_audio, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    print(f"\n  whisper-tiny FGSM (epsilon={attack.epsilon}):")
    print(f"  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")
    print(f"  L_inf:     {linf:.4f}")

    assert adv_wer > clean_wer, (
        f"FGSM did not degrade Whisper-tiny: clean WER {clean_wer:.3f}, "
        f"adv WER {adv_wer:.3f}"
    )
