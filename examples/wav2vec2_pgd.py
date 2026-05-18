"""PGD (epsilon=0.02, 10 steps) vs wav2vec2-base-960h on LibriSpeech-dummy.

White-box iterative attack on a CTC ASR. Each step is a small gradient ascent
in the L-infinity ball of radius epsilon around the clean waveform; after the
step we project back into the ball. Substantially stronger than FGSM at the
same budget: same epsilon, but iterative refinement places energy more
selectively, so the WER goes much higher *and* the perturbation is typically
quieter.

Run with::

    uv run python examples/wav2vec2_pgd.py
"""

from __future__ import annotations

from pathlib import Path

from asr_attack import Attack, run_benchmark


def main() -> None:
    report = run_benchmark(
        model="facebook/wav2vec2-base-960h",
        attack=Attack.pgd(epsilon=0.02, alpha=0.002, n_steps=10),
        dataset="hf-internal-testing/librispeech_asr_dummy",
        n_samples=10,
        split="validation",
        config="clean",
    )

    print(report.summary())

    out_dir = Path(__file__).parent
    report.to_html(out_dir / "wav2vec2_pgd.html")
    report.to_json(out_dir / "wav2vec2_pgd.json")


if __name__ == "__main__":
    main()
