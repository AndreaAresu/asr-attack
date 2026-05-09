# asr-attack

Adversarial robustness toolkit for Hugging Face ASR models (Whisper, wav2vec2,
MMS, ...). Apply attacks like FGSM, PGD, noise injection, and environmental
degradation, then measure the impact with WER/CER on standard datasets.

> **Status:** early alpha. The public API is in place but most attacks are
> stubs that raise `NotImplementedError`.

## Install

```bash
uv add asr-attack       # or: pip install asr-attack
```

## Quickstart

```python
from asr_attack import Attack, HFASRModel, run_benchmark

model = HFASRModel.from_pretrained("openai/whisper-tiny")
attack = Attack.fgsm(epsilon=0.01)

report = run_benchmark(model, attack, "common_voice_17", n_samples=100)
print(report.summary())
```

## Development

```bash
uv sync --group dev
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # full suite (downloads Whisper-tiny + sample)
```

## License

MIT — see [LICENSE](LICENSE).
