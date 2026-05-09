"""Orchestrator: run an attack across a dataset and produce a Report."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from asr_attack.attacks.base import Attack
from asr_attack.models.hf_wrapper import HFASRModel
from asr_attack.report import Report

DatasetLike = str | Iterable[tuple[Any, int, str]]


def run_benchmark(
    model: HFASRModel | str,
    attack: Attack,
    dataset: DatasetLike,
    n_samples: int | None = None,
    split: str = "test",
    batch_size: int = 1,
    seed: int = 0,
) -> Report:
    """Evaluate an ASR model under an adversarial `attack` over `dataset`.

    Args:
        model: a wrapped `HFASRModel` or a HF model id loaded automatically.
        attack: the `Attack` applied to each input waveform.
        dataset: a Hugging Face dataset name (e.g. ``"common_voice_17"``) or an
            iterable yielding ``(audio, sample_rate, reference)`` tuples.
        n_samples: maximum samples to evaluate; ``None`` means the whole split.
        split: dataset split when `dataset` is a string.
        batch_size: model batch size during inference.
        seed: RNG seed for reproducibility.

    Returns:
        A `Report` containing per-sample results and aggregate metrics.
    """
    raise NotImplementedError("run_benchmark is not implemented yet.")
