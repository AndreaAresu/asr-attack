"""Benchmark Report: summary stats plus HTML/Matplotlib output."""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


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
    """Aggregated results of a benchmark run.

    ``clean_wer`` and ``adversarial_wer`` are *pooled* WERs (a single edit-
    distance computation over the concatenation of all references and
    hypotheses), not the mean of per-sample WERs — the standard convention
    in ASR evaluation.
    """

    model_id: str
    attack_name: str
    dataset: str
    samples: list[SampleResult] = field(default_factory=list)
    clean_wer: float = 0.0
    adversarial_wer: float = 0.0
    n_samples: int = 0

    @property
    def wer_delta(self) -> float:
        """Adversarial WER minus clean WER (positive = attack degraded the model)."""
        return self.adversarial_wer - self.clean_wer

    @property
    def mean_snr_db(self) -> float | None:
        """Mean perturbation SNR across samples that report one."""
        snrs = [s.perturbation_db for s in self.samples if s.perturbation_db is not None]
        if not snrs:
            return None
        return float(sum(snrs) / len(snrs))

    def summary(self) -> str:
        """One-screen text summary suitable for terminal printing."""
        lines = [
            "asr-attack benchmark report",
            "===========================",
            f"Model        : {self.model_id}",
            f"Attack       : {self.attack_name}",
            f"Dataset      : {self.dataset}",
            f"Samples      : {self.n_samples}",
            "",
            f"Clean WER    : {self.clean_wer:.3f}",
            f"Adv   WER    : {self.adversarial_wer:.3f}",
            f"WER delta    : {self.wer_delta:+.3f}",
        ]
        if self.mean_snr_db is not None:
            lines.append(f"Mean SNR     : {self.mean_snr_db:.1f} dB")
        return "\n".join(lines) + "\n"

    def to_json(self, path: str | Path) -> Path:
        """Serialize the report to a JSON file. Returns the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def to_html(self, path: str | Path) -> Path:
        """Render an HTML report (with embedded matplotlib charts) to ``path``.

        Charts are PNG, base64-embedded so the HTML is fully self-contained
        (one file, no external assets — easy to email/serve).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_html())
        return path

    def _render_html(self) -> str:
        chart_wer = _wer_distribution_chart(self.samples)
        chart_snr = _snr_vs_wer_chart(self.samples)
        rows_html = "".join(_sample_row_html(i, s) for i, s in enumerate(self.samples))

        mean_snr_str = f"{self.mean_snr_db:.1f} dB" if self.mean_snr_db is not None else "—"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>asr-attack report — {html.escape(self.model_id)} × {html.escape(self.attack_name)}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1100px;
  margin: 2em auto;
  padding: 0 1em;
  color: #222;
}}
h1 {{ font-size: 1.6em; margin-bottom: 0.2em; }}
h2 {{ font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
.meta {{ color: #666; margin-bottom: 1em; font-size: 0.95em; }}
.meta code {{ background: #f0f0f3; padding: 0.1em 0.4em; border-radius: 3px; }}
.kpi {{ display: flex; gap: 1em; margin: 1.5em 0; }}
.kpi div {{ flex: 1; padding: 1em; background: #f7f7fa; border-radius: 6px; }}
.kpi .label {{ color: #777; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi .value {{ font-size: 1.7em; font-weight: 600; margin-top: 0.2em; }}
.charts {{ display: flex; gap: 1em; margin: 1.5em 0; flex-wrap: wrap; }}
.charts img {{ flex: 1; min-width: 350px; max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
th, td {{ border-bottom: 1px solid #eee; padding: 0.5em; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f3; font-weight: 600; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.text {{ max-width: 280px; word-wrap: break-word; }}
</style>
</head>
<body>
<h1>asr-attack benchmark report</h1>
<div class="meta">
  Model: <code>{html.escape(self.model_id)}</code> &middot;
  Attack: <code>{html.escape(self.attack_name)}</code> &middot;
  Dataset: <code>{html.escape(self.dataset)}</code> &middot;
  Samples: {self.n_samples}
</div>

<div class="kpi">
  <div><div class="label">Clean WER</div><div class="value">{self.clean_wer:.3f}</div></div>
  <div><div class="label">Adv. WER</div><div class="value">{self.adversarial_wer:.3f}</div></div>
  <div><div class="label">&Delta; WER</div><div class="value">{self.wer_delta:+.3f}</div></div>
  <div><div class="label">Mean SNR</div><div class="value">{mean_snr_str}</div></div>
</div>

<h2>Distributions</h2>
<div class="charts">
  <img src="data:image/png;base64,{chart_wer}" alt="WER distribution">
  <img src="data:image/png;base64,{chart_snr}" alt="SNR vs WER delta">
</div>

<h2>Per-sample results</h2>
<table>
<thead>
<tr><th>#</th><th>Reference</th><th>Clean</th><th>Adversarial</th><th>Clean WER</th><th>Adv WER</th><th>SNR (dB)</th></tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>
"""


def _sample_row_html(index: int, s: SampleResult) -> str:
    snr_str = f"{s.perturbation_db:.1f}" if s.perturbation_db is not None else "—"
    return (
        f"<tr>"
        f"<td class='num'>{index}</td>"
        f"<td class='text'>{html.escape(s.reference)}</td>"
        f"<td class='text'>{html.escape(s.clean_hypothesis)}</td>"
        f"<td class='text'>{html.escape(s.adversarial_hypothesis)}</td>"
        f"<td class='num'>{s.clean_wer:.3f}</td>"
        f"<td class='num'>{s.adversarial_wer:.3f}</td>"
        f"<td class='num'>{snr_str}</td>"
        f"</tr>"
    )


def _wer_distribution_chart(samples: list[SampleResult]) -> str:
    """Histogram of clean vs adversarial WER, base64-encoded PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    if samples:
        clean = [s.clean_wer for s in samples]
        adv = [s.adversarial_wer for s in samples]
        upper = max(max(clean), max(adv), 1.0) + 0.05
        bins = np.linspace(0.0, upper, 16)
        ax.hist(clean, bins=bins, alpha=0.6, label="clean", color="#3a7", edgecolor="black", linewidth=0.4)
        ax.hist(adv, bins=bins, alpha=0.6, label="adversarial", color="#c44", edgecolor="black", linewidth=0.4)
        ax.legend(loc="upper right")
    ax.set_xlabel("WER")
    ax.set_ylabel("Sample count")
    ax.set_title("Per-sample WER distribution")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _snr_vs_wer_chart(samples: list[SampleResult]) -> str:
    """Scatter of perturbation SNR vs WER delta, base64-encoded PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    snrs = [s.perturbation_db for s in samples if s.perturbation_db is not None]
    deltas = [
        s.adversarial_wer - s.clean_wer for s in samples if s.perturbation_db is not None
    ]
    if snrs:
        ax.scatter(snrs, deltas, color="#c44", alpha=0.7, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="#888", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Perturbation SNR (dB)  —  higher = quieter perturbation")
    ax.set_ylabel("Δ WER  (adv − clean)")
    ax.set_title("Perturbation SNR vs WER degradation")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
