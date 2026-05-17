"""Tests for black-box attacks: NoiseAttack and EnvironmentAttack.

Black-box attacks need no gradient and no model, so the fast unit tests are
purely about correctness of the perturbation operator (shape, dtype,
SNR target, clipping). End-to-end WER tests are marked ``slow`` and use
wav2vec2-base as the target.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")

from asr_attack import Attack
from asr_attack.attacks.environment import EnvironmentAttack
from asr_attack.attacks.noise import NoiseAttack
from asr_attack.metrics.wer import compute_wer
from asr_attack.models.hf_wrapper import HFASRModel

WAV2VEC2_ID = "facebook/wav2vec2-base-960h"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"


@pytest.fixture(scope="module")
def wav2vec2() -> HFASRModel:
    return HFASRModel.from_pretrained(WAV2VEC2_ID, device="cpu")


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
# NoiseAttack
# ---------------------------------------------------------------------------


def test_noise_attack_object_attributes() -> None:
    attack = Attack.noise(snr_db=10.0, kind="uniform")
    assert isinstance(attack, NoiseAttack)
    assert attack.snr_db == 10.0
    assert attack.kind == "uniform"
    assert attack.name == "noise"


def test_noise_attack_preserves_shape_and_dtype() -> None:
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    attack = NoiseAttack(snr_db=10.0, seed=0)
    y = attack.perturb(x, sample_rate=16000)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_noise_attack_hits_target_snr_within_tolerance() -> None:
    """The noise we add should give the requested SNR, modulo float and clipping."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000).astype(np.float32) * 0.05  # ~ -26 dBFS
    for target_snr in [20.0, 10.0, 0.0, -5.0]:
        attack = NoiseAttack(snr_db=target_snr, seed=0)
        y = attack.perturb(x, sample_rate=16000)
        noise = y - x
        sig_p = float(np.mean(x**2))
        noise_p = float(np.mean(noise**2))
        measured_snr = 10.0 * float(np.log10(sig_p / noise_p))
        assert abs(measured_snr - target_snr) < 0.5, (
            f"target SNR {target_snr}, measured {measured_snr:.2f}"
        )


def test_noise_attack_clamps_to_unit_range() -> None:
    x = np.full(4000, 0.95, dtype=np.float32)
    attack = NoiseAttack(snr_db=-10.0, seed=0)  # very loud noise to force clipping
    y = attack.perturb(x, sample_rate=16000)
    assert y.max() <= 1.0
    assert y.min() >= -1.0


def test_noise_attack_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        NoiseAttack(snr_db=10.0, kind="pink").perturb(
            np.zeros(1000, dtype=np.float32), sample_rate=16000
        )


def test_noise_attack_silent_input_is_noop() -> None:
    x = np.zeros(16000, dtype=np.float32)
    y = NoiseAttack(snr_db=10.0, seed=0).perturb(x, sample_rate=16000)
    assert np.array_equal(x, y)


@pytest.mark.slow
def test_noise_attack_degrades_wav2vec2(
    wav2vec2: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """Loud Gaussian noise should drive wav2vec2-base's WER up."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    clean_hyp = wav2vec2.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = NoiseAttack(snr_db=0.0, seed=0)  # signal and noise at equal power
    adv = attack.perturb(audio, sample_rate=sr)

    adv_hyp = wav2vec2.transcribe(adv, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    print(f"\n  wav2vec2 + Gaussian noise (snr_db=0):")
    print(f"  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")

    assert adv_wer > clean_wer


# ---------------------------------------------------------------------------
# EnvironmentAttack
# ---------------------------------------------------------------------------


def _synth_exp_decay_ir(sample_rate: int, duration_s: float, tau_s: float) -> np.ndarray:
    """Toy reverb IR: short exponential decay. Not realistic but enough for tests."""
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    ir = np.exp(-t / tau_s).astype(np.float32)
    # Spike at 0 (direct sound) + tail (early reflections + diffuse).
    return ir


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr)


def test_environment_attack_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="ir_path"):
        EnvironmentAttack().perturb(np.zeros(1000, dtype=np.float32), sample_rate=16000)


def test_environment_attack_factory_returns_dataclass() -> None:
    attack = Attack.environment(ir_path="/tmp/x.wav", background_path=None, snr_db=12.0)
    assert isinstance(attack, EnvironmentAttack)
    assert attack.snr_db == 12.0
    assert attack.ir_path == "/tmp/x.wav"


def test_environment_ir_only_preserves_loudness(tmp_path: Path) -> None:
    """IR-only attack should keep RMS roughly stable (we normalize for it)."""
    sr = 16000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr * 2) * 0.1).astype(np.float32)
    ir = _synth_exp_decay_ir(sr, duration_s=0.2, tau_s=0.05)
    ir_path = tmp_path / "ir.wav"
    _write_wav(ir_path, ir, sr)

    attack = EnvironmentAttack(ir_path=str(ir_path))
    y = attack.perturb(x, sample_rate=sr)

    assert y.shape == x.shape
    assert y.dtype == np.float32
    in_rms = float(np.sqrt(np.mean(x**2)))
    out_rms = float(np.sqrt(np.mean(y**2)))
    assert 0.5 * in_rms <= out_rms <= 1.5 * in_rms


def test_environment_background_only_hits_target_snr(tmp_path: Path) -> None:
    sr = 16000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr * 2) * 0.1).astype(np.float32)
    bg = (rng.standard_normal(sr) * 0.2).astype(np.float32)  # shorter, will loop
    bg_path = tmp_path / "bg.wav"
    _write_wav(bg_path, bg, sr)

    target_snr = 5.0
    attack = EnvironmentAttack(background_path=str(bg_path), snr_db=target_snr)
    y = attack.perturb(x, sample_rate=sr)

    assert y.shape == x.shape
    noise = y - x
    sig_p = float(np.mean(x**2))
    noise_p = float(np.mean(noise**2))
    measured_snr = 10.0 * float(np.log10(sig_p / noise_p))
    assert abs(measured_snr - target_snr) < 0.5


def test_environment_combined_ir_plus_background(tmp_path: Path) -> None:
    sr = 16000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr * 2) * 0.1).astype(np.float32)
    ir = _synth_exp_decay_ir(sr, duration_s=0.15, tau_s=0.04)
    bg = (rng.standard_normal(sr * 3) * 0.2).astype(np.float32)
    ir_path = tmp_path / "ir.wav"
    bg_path = tmp_path / "bg.wav"
    _write_wav(ir_path, ir, sr)
    _write_wav(bg_path, bg, sr)

    attack = EnvironmentAttack(
        ir_path=str(ir_path), background_path=str(bg_path), snr_db=10.0
    )
    y = attack.perturb(x, sample_rate=sr)

    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert y.max() <= 1.0
    assert y.min() >= -1.0


def test_environment_resamples_ir_with_different_sample_rate(tmp_path: Path) -> None:
    """IR at 8 kHz should be resampled to match the input."""
    sr = 16000
    ir_sr = 8000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr * 1) * 0.1).astype(np.float32)
    ir = _synth_exp_decay_ir(ir_sr, duration_s=0.1, tau_s=0.03)
    ir_path = tmp_path / "ir_8k.wav"
    _write_wav(ir_path, ir, ir_sr)

    attack = EnvironmentAttack(ir_path=str(ir_path))
    y = attack.perturb(x, sample_rate=sr)
    assert y.shape == x.shape


@pytest.mark.slow
def test_environment_attack_degrades_wav2vec2(
    tmp_path: Path,
    wav2vec2: HFASRModel,
    librispeech_sample: dict,
    capsys: pytest.CaptureFixture,
) -> None:
    """Reverb + loud background noise should push wav2vec2-base WER up."""
    audio = librispeech_sample["array"]
    sr = librispeech_sample["sampling_rate"]
    reference = librispeech_sample["text"]

    rng = np.random.default_rng(0)
    ir = _synth_exp_decay_ir(sr, duration_s=0.3, tau_s=0.08)
    bg = (rng.standard_normal(sr * 5) * 0.2).astype(np.float32)
    ir_path = tmp_path / "ir.wav"
    bg_path = tmp_path / "bg.wav"
    _write_wav(ir_path, ir, sr)
    _write_wav(bg_path, bg, sr)

    clean_hyp = wav2vec2.transcribe(audio, sample_rate=sr).text
    clean_wer = compute_wer([reference], [clean_hyp])

    attack = EnvironmentAttack(
        ir_path=str(ir_path), background_path=str(bg_path), snr_db=0.0
    )
    adv = attack.perturb(audio, sample_rate=sr)

    adv_hyp = wav2vec2.transcribe(adv, sample_rate=sr).text
    adv_wer = compute_wer([reference], [adv_hyp])

    print(f"\n  wav2vec2 + Environment (synthetic IR + Gaussian bg, snr_db=0):")
    print(f"  reference: {reference}")
    print(f"  clean:     {clean_hyp}")
    print(f"  adv:       {adv_hyp}")
    print(f"  clean WER: {clean_wer:.3f}")
    print(f"  adv WER:   {adv_wer:.3f}")

    assert adv_wer > clean_wer
