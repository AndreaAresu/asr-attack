"""FGSM (epsilon=0.02) vs Whisper-tiny on LibriSpeech-dummy validation (clean).

White-box, single-step attack on a seq2seq ASR. The gradient flows from the
cross-entropy loss back through Whisper's encoder-decoder and the torch-side
log-mel extractor, all the way to the waveform. One forward + one backward
pass per sample. Cheap, but the perturbation typically lands ~10 dB below the
signal — almost imperceptible.

Run with::

    uv run python examples/whisper_tiny_fgsm.py
"""

from __future__ import annotations

from pathlib import Path

from asr_attack import Attack, run_benchmark


def main() -> None:
    report = run_benchmark(
        model="openai/whisper-tiny",
        attack=Attack.fgsm(epsilon=0.02),
        dataset="hf-internal-testing/librispeech_asr_dummy",
        n_samples=10,
        split="validation",
        config="clean",
    )

    print(report.summary())

    out_dir = Path(__file__).parent
    report.to_html(out_dir / "whisper_tiny_fgsm.html")
    report.to_json(out_dir / "whisper_tiny_fgsm.json")


if __name__ == "__main__":
    main()
