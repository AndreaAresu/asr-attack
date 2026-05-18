"""Gaussian noise at SNR=10 dB vs HuBERT-large-ft on LibriSpeech-dummy.

Black-box attack: no gradient, no model required, works on any ASR. We mix
i.i.d. Gaussian noise into the waveform so that the resulting SNR matches the
target. 10 dB is the classic "noisy office" reference point — speech is still
intelligible to humans, but ASR error rates rise noticeably.

HuBERT lives in the wav2vec2 family (CTC over raw waveform) but is a fully
distinct pretrained model — useful to confirm that the attack is genuinely
black-box and not coupled to one architecture.

Run with::

    uv run python examples/hubert_noise.py
"""

from __future__ import annotations

from pathlib import Path

from asr_attack import Attack, run_benchmark


def main() -> None:
    report = run_benchmark(
        model="facebook/hubert-large-ls960-ft",
        attack=Attack.noise(snr_db=10.0, kind="gaussian"),
        dataset="hf-internal-testing/librispeech_asr_dummy",
        n_samples=10,
        split="validation",
        config="clean",
    )

    print(report.summary())

    out_dir = Path(__file__).parent
    report.to_html(out_dir / "hubert_noise.html")
    report.to_json(out_dir / "hubert_noise.json")


if __name__ == "__main__":
    main()
