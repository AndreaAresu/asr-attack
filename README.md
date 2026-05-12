# asr-attack

Adversarial robustness toolkit for Hugging Face ASR models (Whisper, wav2vec2,
MMS, ...). Apply attacks like FGSM, PGD, noise injection, and environmental
degradation, then measure the impact with WER/CER on standard datasets.

> **Status:** early alpha. FGSM is implemented end-to-end against the wav2vec2
> family of CTC models and against Whisper (via a torch-side log-mel
> extractor). PGD, noise, and environment attacks are stubs that raise
> `NotImplementedError`. `run_benchmark` is also a stub.

## Install

```bash
uv add asr-attack       # or: pip install asr-attack
```

## Quickstart (what works today)

```python
import numpy as np
from asr_attack import Attack, HFASRModel
from asr_attack.metrics.wer import compute_wer

model = HFASRModel.from_pretrained("facebook/wav2vec2-base-960h")
audio: np.ndarray = ...   # 1-D float32 waveform in [-1, 1]
reference: str = "..."    # ground-truth transcription

clean_hyp = model.transcribe(audio, sample_rate=16000).text

attack = Attack.fgsm(epsilon=0.02)
adv_audio = attack.perturb(audio, sample_rate=16000, model=model)
adv_hyp = model.transcribe(adv_audio, sample_rate=16000).text

print("clean WER:", compute_wer([reference], [clean_hyp]))
print("adv   WER:", compute_wer([reference], [adv_hyp]))
```

## Attack and model support

White-box attacks (FGSM, PGD) need a differentiable path from the waveform
to the loss. The wrapper exposes `model.supports_waveform_gradient`, used to
gate these attacks.

| Model family | Transcribe | Waveform gradient (FGSM/PGD) |
|---|---|---|
| wav2vec2 / wav2vec2-conformer | ✓ | ✓ |
| HuBERT, WavLM, UniSpeech(-SAT), SEW(-D), data2vec-audio | ✓ | ✓ |
| MMS | ✓ | ✓ |
| Whisper (tiny / base / small / medium / large / large-v2 / large-v3) | ✓ | ✓ (torch-side log-mel) |
| M-CTC-T | ✓ | ✗ (mel-spec input, no torch-side extractor yet) |
| SpeechT5, S2T (seq2seq, non-Whisper) | ✓ | ✗ (no torch-side extractor yet) |

Black-box attacks (noise, environment) work for every model in the table,
once implemented.

## Development

```bash
uv sync --group dev
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # full suite (downloads Whisper-tiny + wav2vec2-base + a LibriSpeech sample)
```

## License

MIT — see [LICENSE](LICENSE).
