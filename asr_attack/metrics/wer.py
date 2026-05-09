"""Word- and character-level error rate metrics, backed by jiwer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer

# Shared text normalization for ASR scoring: case-fold, strip punctuation,
# collapse whitespace. Without this, a model writing "Mr." vs reference
# "MISTER" or trailing punctuation inflates WER for purely cosmetic reasons.
_WORD_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

_CHAR_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


@dataclass
class ErrorRates:
    """Aggregate error rates over a set of (reference, hypothesis) pairs."""

    wer: float
    cer: float
    n_samples: int


def _validate(references: Sequence[str], hypotheses: Sequence[str]) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses have different lengths: "
            f"{len(references)} vs {len(hypotheses)}"
        )
    if len(references) == 0:
        raise ValueError("references and hypotheses must be non-empty")


def compute_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Word Error Rate across paired references and hypotheses."""
    _validate(references, hypotheses)
    return float(
        jiwer.wer(
            reference=list(references),
            hypothesis=list(hypotheses),
            reference_transform=_WORD_TRANSFORM,
            hypothesis_transform=_WORD_TRANSFORM,
        )
    )


def compute_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Character Error Rate across paired references and hypotheses."""
    _validate(references, hypotheses)
    return float(
        jiwer.cer(
            reference=list(references),
            hypothesis=list(hypotheses),
            reference_transform=_CHAR_TRANSFORM,
            hypothesis_transform=_CHAR_TRANSFORM,
        )
    )


def compute_error_rates(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> ErrorRates:
    """Return WER, CER, and sample count in a single call."""
    _validate(references, hypotheses)
    return ErrorRates(
        wer=compute_wer(references, hypotheses),
        cer=compute_cer(references, hypotheses),
        n_samples=len(references),
    )
