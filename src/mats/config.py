"""Load the locked experiment design from configs/defaults.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    dataset: str
    n_prompts: int
    correct_threshold: float
    cue_kind: str
    k_sentence: int
    k_baseline: int
    dedup_cosine_max: float
    monitor_k: int
    n_positions: int
    max_workers: int
    bias_letter: str   # few-shot mode: the letter every worked example answers
    n_few_shot: int    # few-shot mode: number of worked examples in the preamble


def load_config(path: str | Path) -> Config:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config(
        dataset=data["data"]["dataset"],
        n_prompts=int(data["data"]["n_prompts"]),
        correct_threshold=float(data["data"]["correct_threshold"]),
        cue_kind=data["cue"]["kind"],
        k_sentence=int(data["resample"]["k_sentence"]),
        k_baseline=int(data["resample"]["k_baseline"]),
        dedup_cosine_max=float(data["resample"]["dedup_cosine_max"]),
        monitor_k=int(data["monitor"]["k"]),
        n_positions=int(data["resample"]["n_positions"]),
        max_workers=int(data["resample"]["max_workers"]),
        bias_letter=data["cue"]["bias_letter"],
        n_few_shot=int(data["cue"]["n_few_shot"]),
    )
