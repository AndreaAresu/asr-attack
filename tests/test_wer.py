"""Unit tests for asr_attack.metrics.wer."""

from __future__ import annotations

import pytest

from asr_attack.metrics.wer import (
    ErrorRates,
    compute_cer,
    compute_error_rates,
    compute_wer,
)


def test_compute_wer_perfect_match() -> None:
    assert compute_wer(["hello world"], ["hello world"]) == 0.0


def test_compute_wer_normalizes_case_and_punctuation() -> None:
    assert compute_wer(["hello world"], ["Hello, world!"]) == 0.0


def test_compute_wer_substitution() -> None:
    assert compute_wer(["hello world"], ["hello there"]) == pytest.approx(0.5)


def test_compute_wer_multi_sample_aggregate() -> None:
    refs = ["the quick brown fox", "lazy dog"]
    hyps = ["the quick brown fox", "lazy cat"]
    # 1 substitution out of 6 total reference words.
    assert compute_wer(refs, hyps) == pytest.approx(1 / 6)


def test_compute_cer_perfect_match() -> None:
    assert compute_cer(["hello"], ["hello"]) == 0.0


def test_compute_cer_substitution() -> None:
    # "hello" -> "jello": 1 char substitution out of 5 reference chars.
    assert compute_cer(["hello"], ["jello"]) == pytest.approx(0.2)


def test_compute_error_rates_returns_all_three_fields() -> None:
    rates = compute_error_rates(["hello world"], ["hello there"])
    assert isinstance(rates, ErrorRates)
    assert rates.wer == pytest.approx(0.5)
    assert rates.cer > 0.0
    assert rates.n_samples == 1


def test_lengths_must_match() -> None:
    with pytest.raises(ValueError):
        compute_wer(["a", "b"], ["a"])


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compute_wer([], [])
