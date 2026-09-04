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
    )
