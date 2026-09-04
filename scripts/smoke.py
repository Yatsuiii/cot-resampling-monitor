"""End-to-end pipeline check on a stubbed model: no network, no GPU.

    python scripts/smoke.py --backend dummy

Runs the full cue-flip vs control experiment against DummyBackend, writes
outputs/metrics.json and outputs/roc.png, and prints the AUC table.

The dummy models a *cleanly localized* cue (one pivot sentence states the
conclusion), so only the entropy detector is expected to separate the classes
here - the point of the smoke test is that every code path runs and produces a
well-formed ROC, not that the diffuse signal wins on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mats.backend import DummyBackend
from mats.config import load_config
from mats.embed import HashEmbedder
from mats.experiment import run_experiment
from mats.prompts import Question
from mats.report import plot_roc, summarize

REPO = Path(__file__).resolve().parent.parent
_OPTIONS = {"A": "first option", "B": "second option", "C": "third option", "D": "fourth option"}


def synthetic_questions(n: int) -> list[Question]:
    golds = ["A", "B", "C", "D"]
    return [
        Question(
            qid=f"syn-{i:02d}",
            stem=f"Synthetic question number {i}. Which option is correct?",
            options=dict(_OPTIONS),
            gold=golds[i % len(golds)],
        )
        for i in range(n)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["dummy"], default="dummy")
    parser.add_argument("--n", type=int, default=12, help="synthetic questions")
    parser.add_argument("--config", default=str(REPO / "configs" / "defaults.toml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Small rollout counts keep the smoke run to a few seconds.
    cfg = cfg.__class__(**{**cfg.__dict__, "k_sentence": 16, "k_baseline": 16, "monitor_k": 3})

    records = run_experiment(
        DummyBackend(cue_strength=0.7),
        HashEmbedder(),
        synthetic_questions(args.n),
        cfg,
        groundtruth_marker=True,
    )

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    summary = summarize(records)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_roc(records, out / "roc.png")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'metrics.json'} and {out / 'roc.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
