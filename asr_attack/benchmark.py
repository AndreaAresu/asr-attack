"""Orchestrator: run an attack across a dataset and produce a Report."""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from asr_attack.attacks.base import Attack
from asr_attack.metrics.wer import compute_wer
from asr_attack.models.hf_wrapper import HFASRModel
from asr_attack.report import Report, SampleResult

# Each sample is a (waveform, sample_rate, reference_text) triple.
SampleTuple = tuple[np.ndarray, int, str]
DatasetLike = str | Iterable[SampleTuple]


def _benchmark_default_device() -> str:
    """CUDA if available, else CPU. We deliberately skip MPS as a default:
    PyTorch on MPS does not implement ``aten::_ctc_loss``, which would crash
    any FGSM/PGD run on a CTC model (wav2vec2 family). Users on Mac who only
    need transcription / black-box attacks can pass ``device="mps"`` explicitly.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_benchmark(
    model: HFASRModel | str,
    attack: Attack,
    dataset: DatasetLike,
    n_samples: int | None = None,
    split: str = "test",
    config: str | None = None,
    text_column: str = "text",
    audio_column: str = "audio",
    device: str | None = None,
    verbose: bool = True,
) -> Report:
    """Evaluate an ASR model under an adversarial ``attack`` over ``dataset``.

    Args:
        model: a loaded ``HFASRModel`` or a Hugging Face model id string
            (loaded automatically via ``HFASRModel.from_pretrained``).
        attack: the ``Attack`` instance to apply to each sample.
        dataset: a Hugging Face dataset id (e.g.
            ``"hf-internal-testing/librispeech_asr_dummy"``) or an iterable
            yielding ``(audio_np, sample_rate_int, reference_text)`` triples.
        n_samples: maximum samples to evaluate; ``None`` means iterate the
            entire split (dangerous for full Common Voice / LibriSpeech —
            those are huge).
        split: dataset split when ``dataset`` is a string.
        config: dataset config when ``dataset`` is a string (e.g. ``"clean"``
            for LibriSpeech, ``"en"`` for Common Voice).
        text_column: name of the column holding the reference transcription.
            Defaults to ``"text"`` (LibriSpeech). Common Voice uses
            ``"sentence"``.
        audio_column: name of the audio column. Almost always ``"audio"`` on
            modern HF datasets.
        device: device to load the model on (only used when ``model`` is a
            string). Defaults to ``"cuda"`` if available else ``"cpu"``.
            **MPS is intentionally never the default**: PyTorch on MPS lacks
            ``aten::_ctc_loss``, which would crash FGSM/PGD on CTC models.
            Pass ``device="mps"`` explicitly if you only need transcription
            or black-box attacks.
        verbose: show a tqdm progress bar.

    Returns:
        A ``Report`` with per-sample :class:`SampleResult` entries and
        aggregate pooled WERs (the standard ASR convention — a single
        edit-distance over the concatenation, not the mean of per-sample
        WERs).
    """
    if isinstance(model, str):
        chosen_device = device or _benchmark_default_device()
        if verbose:
            print(f"Loading model {model} on {chosen_device}...")
        model = HFASRModel.from_pretrained(model, device=chosen_device)

    dataset_label = dataset if isinstance(dataset, str) else "<iterable>"
    sample_iter = _resolve_dataset(
        dataset,
        split=split,
        config=config,
        text_column=text_column,
        audio_column=audio_column,
    )
    if n_samples is not None:
        sample_iter = _take(sample_iter, n_samples)

    sample_results: list[SampleResult] = []
    iterator = (
        tqdm(sample_iter, total=n_samples, desc=f"{attack.name} on {model.model_id}")
        if verbose
        else sample_iter
    )

    for audio, sr, reference in iterator:
        sample_results.append(_run_one(model, attack, audio, sr, reference))

    refs = [r.reference for r in sample_results]
    clean_hyps = [r.clean_hypothesis for r in sample_results]
    adv_hyps = [r.adversarial_hypothesis for r in sample_results]

    return Report(
        model_id=model.model_id,
        attack_name=attack.name,
        dataset=dataset_label,
        samples=sample_results,
        clean_wer=compute_wer(refs, clean_hyps) if refs else 0.0,
        adversarial_wer=compute_wer(refs, adv_hyps) if refs else 0.0,
        n_samples=len(sample_results),
    )


def _run_one(
    model: HFASRModel,
    attack: Attack,
    audio: np.ndarray,
    sample_rate: int,
    reference: str,
) -> SampleResult:
    """Transcribe clean, apply attack, transcribe adversarial, score both."""
    clean_hyp = model.transcribe(audio, sample_rate=sample_rate).text
    adv = attack.perturb(audio, sample_rate=sample_rate, model=model)
    adv_hyp = model.transcribe(adv, sample_rate=sample_rate).text

    clean_wer = compute_wer([reference], [clean_hyp])
    adv_wer = compute_wer([reference], [adv_hyp])

    # Perturbation SNR (dB) — None for silent inputs or zero perturbations.
    snr_db: float | None = None
    a = audio.astype(np.float64)
    d = adv.astype(np.float64) - a
    sig_p = float(np.mean(a**2))
    noise_p = float(np.mean(d**2))
    if sig_p > 0.0 and noise_p > 0.0:
        snr_db = float(10.0 * np.log10(sig_p / noise_p))

    return SampleResult(
        reference=reference,
        clean_hypothesis=clean_hyp,
        adversarial_hypothesis=adv_hyp,
        clean_wer=clean_wer,
        adversarial_wer=adv_wer,
        perturbation_db=snr_db,
    )


def _resolve_dataset(
    dataset: DatasetLike,
    split: str,
    config: str | None,
    text_column: str,
    audio_column: str,
) -> Iterator[SampleTuple]:
    """Turn whatever the user gave us into a stream of (audio, sr, text)."""
    if not isinstance(dataset, str):
        yield from dataset
        return

    try:
        import datasets as hf_datasets
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Loading a dataset by name requires the `datasets` package. "
            "Install it via `uv add datasets`."
        ) from e

    # decode=False keeps the raw audio bytes, which we then decode with
    # soundfile — avoids the torchcodec dependency that recent versions of
    # `datasets` would otherwise pull in for auto-decoding.
    ds = hf_datasets.load_dataset(dataset, config, split=split)
    ds = ds.cast_column(audio_column, hf_datasets.Audio(decode=False))

    for row in ds:
        audio_field: Any = row[audio_column]
        audio, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        yield np.asarray(audio, dtype=np.float32), int(sr), str(row[text_column])


def _take(iterator: Iterator[SampleTuple], n: int) -> Iterator[SampleTuple]:
    """Yield at most ``n`` items from ``iterator``. Equivalent to itertools.islice."""
    for i, item in enumerate(iterator):
        if i >= n:
            return
        yield item
