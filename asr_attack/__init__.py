"""asr-attack: adversarial robustness toolkit for Hugging Face ASR models."""

from asr_attack.attacks.base import Attack
from asr_attack.benchmark import run_benchmark
from asr_attack.models.hf_wrapper import HFASRModel
from asr_attack.report import Report

__version__ = "0.1.0"

__all__ = [
    "Attack",
    "HFASRModel",
    "Report",
    "__version__",
    "run_benchmark",
]
