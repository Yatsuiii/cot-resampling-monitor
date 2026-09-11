"""Run an auditable Phase-1 repair experiment on a single Kaggle T4.

The run is deliberately fixed to the v2 cohort and four policies. The only
methodological repair is at the completion boundary: a length-terminated
completion is not allowed to contribute a parsed answer, and every raw sample
is retained in the output for audit.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict


MODEL = "Qwen/Qwen3-4B"
N_GENERATIONS = int(os.environ.get("MATS_N_GENERATIONS", "8"))
MAX_TOKENS_FULL = int(os.environ.get("MATS_MAX_TOKENS_FULL", "4096"))
MAX_TOKENS_CONTINUE = int(os.environ.get("MATS_MAX_TOKENS_CONTINUE", "3000"))
MAX_WORKERS = int(os.environ.get("MATS_MAX_WORKERS", "16"))
N_QUESTIONS = int(os.environ.get("MATS_N_QUESTIONS", "20"))
SEED = 20260908
TRUNCATION_GATE = 0.10


def _wait_ready(url: str = "http://localhost:8000/health", timeout: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM server did not come up")


def _serialize(result) -> dict:
    row = asdict(result)
    row["cue_effect"] = result.cue_effect
    row["residual_pull"] = result.residual_pull
    return row


def _summary(results) -> dict:
    from mats.repair import dose_response

    conditions = ("clean", "cued", "source_removal", "explicit_correction")
    truncation = {
        condition: sum(r.rates[condition].truncated for r in results) / len(results)
        for condition in conditions
    }
    answerable = [r for r in results if r.rates["clean"].gold_rate >= 0.75]
    report = {
        "model": MODEL,
        "n_questions": len(results),
        "n_answerable": len(answerable),
        "n_generations": N_GENERATIONS,
        "max_tokens_full": MAX_TOKENS_FULL,
        "max_tokens_continue": MAX_TOKENS_CONTINUE,
        "max_workers": MAX_WORKERS,
        "truncation_gate": TRUNCATION_GATE,
        "truncation_by_condition": truncation,
        "gate_passed": max(truncation.values(), default=1.0) < TRUNCATION_GATE,
        "mean_tokens_by_condition": {
            condition: sum(r.rates[condition].mean_tokens for r in results) / len(results)
            for condition in conditions
        },
        "mean_rates_all": {
            condition: {
                "cue_rate": sum(r.rates[condition].cue_rate for r in results) / len(results),
                "gold_rate": sum(r.rates[condition].gold_rate for r in results) / len(results),
                "unusable": sum(r.rates[condition].unusable for r in results) / len(results),
            }
            for condition in conditions
        },
    }
    if answerable:
        report["mean_rates_answerable"] = {
            condition: {
                "cue_rate": sum(r.rates[condition].cue_rate for r in answerable) / len(answerable),
                "gold_rate": sum(r.rates[condition].gold_rate for r in answerable) / len(answerable),
                "unusable": sum(r.rates[condition].unusable for r in answerable) / len(answerable),
            }
            for condition in conditions
        }
        report["dose_response_answerable"] = dose_response(answerable)
    return report


def main() -> None:
    # Kaggle mounts this source dataset at a version-dependent path. The kernel
    # wrapper places the repo root in /kaggle/working/MATS before invoking us.
    repo = pathlib.Path("/kaggle/working/MATS")
    if not (repo / "src" / "mats").is_dir():
        raise RuntimeError(f"missing source package at {repo / 'src' / 'mats'}")
    sys.path.insert(0, str(repo / "src"))

    from transformers import AutoTokenizer
    from mats.backend_vllm import VLLMBackend
    from mats.data import load_arc_challenge
    from mats.prompts import cue_target
    from mats.repair import run_question

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--dtype",
            "float16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "6144",
            "--gpu-memory-utilization",
            "0.92",
        ]
    )
    atexit.register(server.terminate)
    _wait_ready()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def render(prompt: str, prefix: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + prefix

    backend = VLLMBackend(
        model=MODEL,
        temperature=0.8,
        top_p=0.95,
        stop=["<|im_end|>"],
        render=render,
    )

    qid_path = repo / "configs" / "repair_phase1_qids.txt"
    qids = [
        line.strip()
        for line in qid_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pool = {q.qid: q for q in load_arc_challenge("validation")}
    missing = sorted(set(qids) - set(pool))
    if missing:
        raise RuntimeError(f"cohort questions missing from ARC validation: {missing}")
    questions = [pool[qid] for qid in qids[:N_QUESTIONS]]

    output = pathlib.Path("/kaggle/working/phase1_clean.json")
    results = []
    for index, question in enumerate(questions):
        result = run_question(
            backend,
            question,
            cue_letter=cue_target(question),
            n_gen=N_GENERATIONS,
            seed=SEED + index * 1000,
            max_workers=MAX_WORKERS,
            max_tokens_full=MAX_TOKENS_FULL,
            max_tokens_continue=MAX_TOKENS_CONTINUE,
        )
        results.append(result)
        partial = {"summary": _summary(results), "results": [_serialize(r) for r in results]}
        output.write_text(json.dumps(partial, indent=2))
        print(
            f"[{index + 1:02d}/{len(questions)}] {question.qid} "
            f"cue={result.cue_effect:.3f} residual={result.residual_pull:.3f} "
            f"trunc_source={result.rates['source_removal'].truncated:.3f}",
            flush=True,
        )

    final = {"summary": _summary(results), "results": [_serialize(r) for r in results]}
    output.write_text(json.dumps(final, indent=2))
    print(json.dumps(final["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
