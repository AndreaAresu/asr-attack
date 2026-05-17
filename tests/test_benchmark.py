"""Tests for run_benchmark and the Report class.

Fast tests use a tiny in-memory iterable of synthetic samples and a
fake-loaded HFASRModel (no download) when possible. The single slow test
runs a real noise attack on wav2vec2-base over 3 LibriSpeech samples and
exercises the HF-dataset loading path.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")

from asr_attack import HFASRModel, Report, run_benchmark
from asr_attack.attacks.base import Attack
from asr_attack.attacks.noise import NoiseAttack
from asr_attack.benchmark import _run_one
from asr_attack.report import SampleResult


# ---------------------------------------------------------------------------
# Fast tests — fully synthetic, no model loading, no downloads
# ---------------------------------------------------------------------------


class _NullAttack(Attack):
    """Identity attack: returns the input unchanged. For testing the loop."""

    name = "null"

    def perturb(self, audio, sample_rate, model=None, target=None):
        return np.asarray(audio, dtype=np.float32)


class _DummyModel:
    """Minimal HFASRModel stand-in for the orchestrator tests."""

    model_id = "dummy/identity"
    device = "cpu"
    sample_rate = 16000
    kind = "ctc"
    supports_waveform_gradient = False

    def transcribe(self, audio, sample_rate=None):
        # Always returns a fixed string so WER is deterministic against the
        # synthetic references we construct in the tests.
        from asr_attack.models.hf_wrapper import TranscriptionResult

        return TranscriptionResult(text="hello world", sample_rate=sample_rate or 16000)


def _synthetic_samples(n: int) -> Iterator[tuple[np.ndarray, int, str]]:
    rng = np.random.default_rng(0)
    for _ in range(n):
        audio = (rng.standard_normal(8000) * 0.1).astype(np.float32)
        yield audio, 16000, "hello world"


def test_run_benchmark_on_iterable_returns_report() -> None:
    report = run_benchmark(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        dataset=_synthetic_samples(5),
        verbose=False,
    )
    assert isinstance(report, Report)
    assert report.n_samples == 5
    assert report.model_id == "dummy/identity"
    assert report.attack_name == "null"
    assert report.dataset == "<iterable>"
    # Null attack on a model that always transcribes "hello world" against
    # a reference "hello world" -> both WERs are 0.
    assert report.clean_wer == 0.0
    assert report.adversarial_wer == 0.0
    assert report.wer_delta == 0.0


def test_run_benchmark_respects_n_samples() -> None:
    report = run_benchmark(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        dataset=_synthetic_samples(100),
        n_samples=4,
        verbose=False,
    )
    assert report.n_samples == 4
    assert len(report.samples) == 4


def test_summary_is_non_empty_text() -> None:
    report = run_benchmark(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        dataset=_synthetic_samples(3),
        verbose=False,
    )
    text = report.summary()
    assert isinstance(text, str)
    assert "dummy/identity" in text
    assert "null" in text
    assert "Clean WER" in text


def test_to_json_writes_round_trippable_file(tmp_path: Path) -> None:
    report = run_benchmark(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        dataset=_synthetic_samples(3),
        verbose=False,
    )
    path = tmp_path / "report.json"
    written = report.to_json(path)
    assert written == path
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["model_id"] == "dummy/identity"
    assert payload["n_samples"] == 3
    assert len(payload["samples"]) == 3


def test_to_html_writes_self_contained_file(tmp_path: Path) -> None:
    report = run_benchmark(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        dataset=_synthetic_samples(3),
        verbose=False,
    )
    path = tmp_path / "report.html"
    report.to_html(path)
    body = path.read_text()
    assert "<!DOCTYPE html>" in body
    assert "dummy/identity" in body
    # Charts must be embedded as base64-encoded PNGs (no external assets).
    assert "data:image/png;base64," in body
    # Per-sample table must include each row.
    assert body.count("<tr>") >= 4  # 1 header + 3 rows


def test_run_one_records_perturbation_snr() -> None:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    result = _run_one(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=NoiseAttack(snr_db=10.0, seed=0),
        audio=audio,
        sample_rate=16000,
        reference="hello world",
    )
    assert isinstance(result, SampleResult)
    assert result.perturbation_db is not None
    # NoiseAttack at SNR=10dB should yield a perturbation_db near 10.
    assert abs(result.perturbation_db - 10.0) < 1.0


def test_run_one_silent_input_has_none_snr() -> None:
    result = _run_one(
        model=_DummyModel(),  # type: ignore[arg-type]
        attack=_NullAttack(),
        audio=np.zeros(8000, dtype=np.float32),
        sample_rate=16000,
        reference="hello world",
    )
    assert result.perturbation_db is None


# ---------------------------------------------------------------------------
# Slow test — exercises the HF-dataset string path end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_benchmark_against_librispeech_dummy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """End-to-end: load 3 LibriSpeech samples by HF dataset name, attack with
    NoiseAttack, generate Report + summary + JSON + HTML."""
    report = run_benchmark(
        model="facebook/wav2vec2-base-960h",
        attack=NoiseAttack(snr_db=0.0, seed=0),
        dataset="hf-internal-testing/librispeech_asr_dummy",
        n_samples=3,
        split="validation",
        config="clean",
        text_column="text",
        verbose=False,
    )

    assert report.n_samples == 3
    # Loud (SNR=0) Gaussian noise on wav2vec2 reliably drives WER up over
    # any of these clean samples.
    assert report.adversarial_wer > report.clean_wer
    # Per-sample SNR should be close to 0 dB (target SNR of the attack).
    snrs = [s.perturbation_db for s in report.samples if s.perturbation_db is not None]
    assert snrs
    mean_snr = sum(snrs) / len(snrs)
    assert abs(mean_snr - 0.0) < 1.5

    summary = report.summary()
    print("\n" + summary)
    assert "facebook/wav2vec2-base-960h" in summary

    json_path = report.to_json(tmp_path / "report.json")
    html_path = report.to_html(tmp_path / "report.html")
    assert json_path.exists()
    assert html_path.exists()
    assert html_path.stat().st_size > 1000  # rough sanity check
