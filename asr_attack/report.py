"""Benchmark Report: summary stats plus HTML/Matplotlib output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampleResult:
    """Per-sample outcome of a benchmark run."""

    reference: str
    clean_hypothesis: str
    adversarial_hypothesis: str
    clean_wer: float
    adversarial_wer: float
    perturbation_db: float | None = None


@dataclass
class Report:
    """Aggregated results of a benchmark run."""

    model_id: str
    attack_name: str
    dataset: str
    samples: list[SampleResult] = field(default_factory=list)
    clean_wer: float = 0.0
    adversarial_wer: float = 0.0
    n_samples: int = 0

    def summary(self) -> str:
        """Return a one-screen text summary of clean vs adversarial performance."""
        raise NotImplementedError("Report.summary is not implemented yet.")

    def to_html(self, path: str | Path) -> Path:
        """Render an HTML report with matplotlib charts to `path` and return it."""
        raise NotImplementedError("Report.to_html is not implemented yet.")

    def to_json(self, path: str | Path) -> Path:
        """Serialize the report to JSON at `path` and return the written path."""
        raise NotImplementedError("Report.to_json is not implemented yet.")
