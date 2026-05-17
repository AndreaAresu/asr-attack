"""End-to-end PGD tests: attack wav2vec2-base and Whisper-tiny on a clean
LibriSpeech sample and verify the WER goes up while the L_inf budget holds.

Slow tests download ~360MB of wav2vec2-base + ~150MB of Whisper-tiny plus a
small LibriSpeech sample. Run with: ``uv run pytest -m slow``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")

from asr_attack import Attack
from asr_attack.attacks.pgd import PGDAttack
from asr_attack.metrics.wer import compute_wer
from asr_attack.models.hf_wrapper import HFASRModel

WAV2VEC2_ID = "facebook/wav2vec2-base-960h"
WHISPER_ID = "openai/whisper-tiny"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"


@pytest.fixture(scope="module")
def wav2vec2() -> HFASRModel:
    return HFASRModel.from_pretrained(WAV2VEC2_ID, device="cpu")


@pytest.fixture(scope="module")
def whisper_tiny() -> HFASRModel:
    return HFASRModel.from_pretrained(WHISPER_ID, device="cpu")


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


def test_pgd_attack_object_attributes() -> None:
    attack = Attack.pgd(epsilon=0.05, alpha=0.005, n_steps=10, random_start=False)
    assert isinstance(attack, PGDAttack)
    assert attack.epsilon == 0.05
    assert attack.alpha == 0.005
    assert attack.n_steps == 10
    assert attack.random_start is False
    assert attack.name == "pgd"


def test_pgd_rejects_models_without_waveform_gradient() -> None:
    """Mirror of the FGSM rejection test: PGD must refuse via the same property."""
    fake = HFASRModel(model_id="not-a-real-model", device="cpu")
    assert fake.supports_waveform_gradient is False

    attack = Attack.pgd(epsilon=0.01, alpha=0.005, n_steps=2)
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(NotImplementedError, match="waveform"):
        attack.perturb(audio, sample_rate=16000, model=fake)


@pytest.mark.slow
def test_pgd_increases_wer_on_wav2vec2(
    wav2vec2: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """PGD with multiple steps degrades wav2vec2-base and respects the L_inf budget."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    clean_hyp = wav2vec2.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = PGDAttack(
        epsilon=0.02, alpha=0.005, n_steps=10, random_start=True, seed=0
    )
    adv_audio = attack.perturb(audio, sample_rate=sr, model=wav2vec2)

    assert adv_audio.shape == audio.shape
    assert adv_audio.dtype == np.float32
    linf = float(np.abs(adv_audio - audio).max())
    assert linf <= attack.epsilon + 1e-6, f"L_inf {linf} exceeds epsilon {attack.epsilon}"

    adv_hyp = wav2vec2.transcribe(adv_audio, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    print(
        f"\n  wav2vec2 PGD "
        f"(eps={attack.epsilon}, alpha={attack.alpha}, n_steps={attack.n_steps}):"
    )
    print(f"  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")
    print(f"  L_inf:     {linf:.4f}")

    assert adv_wer > clean_wer, (
        f"PGD did not degrade wav2vec2: clean WER {clean_wer:.3f}, "
        f"adv WER {adv_wer:.3f}"
    )


@pytest.mark.slow
def test_pgd_increases_wer_on_whisper(
    whisper_tiny: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """PGD via the torch-side mel extractor degrades Whisper-tiny.

    n_steps is kept low (5) so the test runs under ~1 minute on CPU. The
    canonical default for research-grade PGD on Whisper is n_steps=40 with
    smaller alpha; that's left for offline experiments.
    """
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    clean_hyp = whisper_tiny.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = PGDAttack(
        epsilon=0.02, alpha=0.008, n_steps=5, random_start=True, seed=0
    )
    adv_audio = attack.perturb(audio, sample_rate=sr, model=whisper_tiny)

    assert adv_audio.shape == audio.shape
    assert adv_audio.dtype == np.float32
    linf = float(np.abs(adv_audio - audio).max())
    assert linf <= attack.epsilon + 1e-6

    adv_hyp = whisper_tiny.transcribe(adv_audio, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    print(
        f"\n  whisper-tiny PGD "
        f"(eps={attack.epsilon}, alpha={attack.alpha}, n_steps={attack.n_steps}):"
    )
    print(f"  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")
    print(f"  L_inf:     {linf:.4f}")

    assert adv_wer > clean_wer
