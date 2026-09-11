"""Run the Phase 1 elicitation sweep on a single Kaggle T4.

The server lifecycle, chat-template rendering and partial-output discipline are
carried over from scripts/repair_clean_kaggle.py, which is already proven on
this hardware. The important one is partial output: a Kaggle session that dies
at hour four must not lose the first three.

Phase 1 asks whether any (cue family, corpus) cell produces flips that the chain
of thought never mentions. The previous run's positive class was empty, so the
detector was never tested and nothing downstream means anything until a
non-empty cell exists.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
import urllib.request

MODEL = os.environ.get("MATS_MODEL", "Qwen/Qwen3-4B")
N_ITEMS = int(os.environ.get("MATS_N_ITEMS", "40"))
MAX_WORKERS = int(os.environ.get("MATS_MAX_WORKERS", "16"))
MAX_TOKENS = int(os.environ.get("MATS_MAX_TOKENS", "4096"))
DATASETS = os.environ.get("MATS_DATASETS", "arc-challenge,mmlu").split(",")
FAMILIES = os.environ.get(
    "MATS_FAMILIES", "authority,sycophancy,metadata,grader,few_shot,positional").split(",")
BIAS_LETTER = os.environ.get("MATS_BIAS_LETTER", "A")
SEED = int(os.environ.get("MATS_SEED", "20260912"))
CORRECT_THRESHOLD = float(os.environ.get("MATS_CORRECT_THRESHOLD", "0.8"))


def _wait_ready(url: str = "http://localhost:8000/health", timeout: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM server did not come up")


def _start_server():
    server = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--dtype", "float16", "--tensor-parallel-size", "1",
        "--max-model-len", "6144", "--gpu-memory-utilization", "0.92",
    ])
    atexit.register(server.terminate)
    _wait_ready()
    return server


def _backend():
    from transformers import AutoTokenizer

    from mats.backend_vllm import VLLMBackend

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def render(prompt: str, prefix: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        return text + prefix

    return VLLMBackend(model=MODEL, temperature=0.8, top_p=0.95,
                       stop=["<|im_end|>"], render=render, max_tokens=MAX_TOKENS)


def sweep(backend, out: pathlib.Path, *, datasets=DATASETS, families=FAMILIES,
          n_items=N_ITEMS, seed=SEED, bias_letter=BIAS_LETTER,
          correct_threshold=CORRECT_THRESHOLD) -> dict:
    """The grid loop. Writes summary.json after EVERY cell, so a killed session
    leaves the cells that finished rather than nothing at all."""
    from mats.cues import family
    from mats.data import keep_answerable
    from mats.datasets import load
    from mats.sweep import manifest, run_cell, summarise, write_traces

    out.mkdir(parents=True, exist_ok=True)
    run = manifest(MODEL, f"n={n_items},seed={seed}", seed)
    state = {"manifest": run, "n_items_per_cell": n_items,
             "datasets": datasets, "families": families,
             "cells": {}, "failed_cells": {}, "complete": False}

    def flush():
        (out / "summary.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    flush()
    for dataset in datasets:
        pool = load(dataset)
        questions = keep_answerable(backend, pool, threshold=correct_threshold,
                                    k=4, seed=seed, limit=n_items)
        print(f"[{dataset}] {len(questions)} answerable of {len(pool)}", flush=True)
        for name in families:
            key = f"{dataset}__{name}"
            try:
                fam = family(name)
                traces = run_cell(backend, questions, fam, dataset=dataset,
                                  model=MODEL, bias_letter=bias_letter, seed=seed)
            except Exception as exc:
                # H30: one unbuildable family must not abort a grid costing
                # GPU-hours. Record it and keep going.
                state["failed_cells"][key] = {
                    "error": str(exc), "traceback": traceback.format_exc(limit=2)}
                print(f"  {key}: FAILED - {exc}", flush=True)
                flush()
                continue
            digest = write_traces(out / f"{key}.jsonl.gz", traces)
            state["cells"][key] = {**summarise(traces, fam),
                                   "traces_sha256": digest,
                                   "traces_file": f"{key}.jsonl.gz"}
            s = state["cells"][key]
            print(f"  {key}: flips {s['n_flipped']}/{s['n_items']}  "
                  f"positive {s['n_positive']}  "
                  f"{'PASSES G-A' if s['passes_G_A'] else 'below gate'}", flush=True)
            flush()

    state["complete"] = True
    state["cells_passing_G_A"] = [k for k, v in state["cells"].items() if v["passes_G_A"]]
    flush()
    return state


def main() -> None:
    repo = pathlib.Path("/kaggle/working/MATS")
    if not (repo / "src" / "mats").is_dir():
        raise RuntimeError(f"missing source package at {repo / 'src' / 'mats'}")
    sys.path.insert(0, str(repo / "src"))

    _start_server()
    state = sweep(_backend(), pathlib.Path("/kaggle/working/phase1"))

    passing = state.get("cells_passing_G_A", [])
    print(f"\n{len(state['cells'])} cells, {len(state['failed_cells'])} failed, "
          f"{len(passing)} passing G-A", flush=True)
    print(json.dumps({k: {kk: v[kk] for kk in
                          ("n_flipped", "n_positive", "passes_G_A")}
                      for k, v in state["cells"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
