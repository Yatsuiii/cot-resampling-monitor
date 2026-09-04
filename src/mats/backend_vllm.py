"""`Backend` implementation talking to a local vLLM server (Kaggle GPU).

Uses the raw `/v1/completions` endpoint, not chat completions, on purpose: the
resampling method prefills part of the reasoning trace and continues from an
arbitrary sentence boundary, which only the raw-prompt endpoint supports
cleanly. The caller is responsible for having already applied the model's chat
template to `prompt` (the Kaggle notebook does this once via the tokenizer).

Determinism: vLLM honours the `seed` field per request, so (prompt, seed) is
reproducible as long as the server config is fixed.

Only the standard library is imported here so the analysis package stays
installable without vLLM present; the server itself runs in the notebook.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable


def _plain(prompt: str, prefix: str) -> str:
    return prompt + prefix


class VLLMBackend:
    """`render(prompt, prefix) -> str` produces the raw text sent to the model.

    The Kaggle notebook passes a `render` that applies the tokenizer's chat
    template to `prompt` as a user turn and appends `prefix` as the start of the
    assistant turn - that is what makes "continue the CoT from sentence i" work.
    The default just concatenates, which is enough for base models and tests.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        timeout: float = 120.0,
        render: Callable[[str, str], str] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/completions"
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.timeout = timeout
        self._render = render or _plain

    def complete(
        self, prompt: str, *, prefix: str = "", seed: int, max_tokens: int = 1024
    ) -> str:
        body = {
            "model": self.model,
            "prompt": self._render(prompt, prefix),
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": seed,
        }
        if self.stop:
            body["stop"] = self.stop
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        return payload["choices"][0]["text"]
