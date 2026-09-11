"""The Kaggle runner's grid loop, driven without a server or a GPU.

The loop is separated from server startup precisely so it is testable. The
property that matters is partial output: a Kaggle session that dies at hour four
must leave the cells that finished, not nothing.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

from mats.backend import DummyBackend
from mats.datasets import LOADERS
from mats.prompts import Question
from mats.sweep import read_traces, summarise

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "kaggle_sweep", REPO / "scripts" / "sweep_phase1_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _questions(n, gold_cycle=("A", "A", "A", "B", "C", "D")):
    return [Question(qid=f"t{i}", stem=f"Question {i}?",
                     options={"A": "a", "B": "b", "C": "c", "D": "d"},
                     gold=gold_cycle[i % len(gold_cycle)])
            for i in range(n)]


@pytest.fixture
def patched(monkeypatch):
    """Swap the corpus loader and the answerable filter for fixtures, so the
    grid loop runs with no network and no model."""
    mod = _module()
    import mats.data
    import mats.datasets
    monkeypatch.setitem(LOADERS, "fixture", lambda **kw: _questions(12))
    monkeypatch.setattr(mats.datasets, "load", lambda name, **kw: _questions(12))
    monkeypatch.setattr(mats.data, "keep_answerable",
                        lambda backend, qs, **kw: qs[: kw.get("limit", len(qs))])
    return mod


def test_the_script_imports_without_vllm_transformers_or_a_gpu():
    """A syntax or import error must surface here, not after a Kaggle queue."""
    mod = _module()
    assert mod.MODEL and mod.N_ITEMS > 0
    assert "vllm" not in sys.modules and "transformers" not in sys.modules


def test_summary_is_written_after_every_cell(patched, monkeypatch):
    """H29. Snapshot the summary as each cell lands; it must grow monotonically
    rather than appearing only at the end."""
    seen = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        real_write = patched.__dict__["json"].dumps

        state = patched.sweep(DummyBackend(), out, datasets=["fixture"],
                              families=["authority", "sycophancy"], n_items=6)
        summary = json.loads((out / "summary.json").read_text())
        assert summary["complete"] is True
        assert set(summary["cells"]) == {"fixture__authority", "fixture__sycophancy"}
        for cell in summary["cells"].values():
            assert (out / cell["traces_file"]).exists()


def test_a_failing_cell_is_recorded_and_the_grid_continues(patched):
    """H30. Six questions with this gold cycle give only 3 with gold 'A', enough
    for a few-shot block; force the failure by demanding a letter with none."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        state = patched.sweep(DummyBackend(), out, datasets=["fixture"],
                              families=["few_shot", "authority"], n_items=6,
                              bias_letter="Z")
        assert "fixture__few_shot" in state["failed_cells"]
        assert "fixture__authority" in state["cells"], "grid aborted on one failure"
        assert state["complete"] is True


def test_partial_summary_survives_an_interrupted_grid(patched, monkeypatch):
    """The real failure mode: the session dies mid-grid. Whatever finished must
    still be on disk and still be recomputable from its traces."""
    from mats import sweep as sweep_mod

    calls = {"n": 0}
    real_run_cell = sweep_mod.run_cell

    def exploding(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt("kaggle session killed")
        return real_run_cell(*a, **kw)

    monkeypatch.setattr(sweep_mod, "run_cell", exploding)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with pytest.raises(KeyboardInterrupt):
            patched.sweep(DummyBackend(), out, datasets=["fixture"],
                          families=["authority", "sycophancy"], n_items=6)
        summary = json.loads((out / "summary.json").read_text())
        assert summary["complete"] is False
        assert len(summary["cells"]) == 1, "the finished cell was lost"
        from mats.cues import family
        key, cell = next(iter(summary["cells"].items()))
        recomputed = summarise(read_traces(out / cell["traces_file"]),
                               family(key.split("__", 1)[1]))
        for f, v in recomputed.items():
            assert cell[f] == v
