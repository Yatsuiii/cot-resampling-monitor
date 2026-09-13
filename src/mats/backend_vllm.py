"""`Backend` implementation talking to a local vLLM server (Kaggle GPU).

Uses the raw `/v1/completions` endpoint, not chat completions, on purpose: the
resampling method prefills part of the reasoning trace and continues from an
arbitrary sentence boundary, which only the raw-prompt endpoint supports
cleanly. The caller is responsible for having already applied the model's chat
template to `prompt` (the Kaggle notebook does this once via the tokenizer).

Determinism: vLLM honours the `seed` field per request, but that does NOT make
(prompt, seed) reproducible under concurrency. The 09-12 grid ran two cue
families that happened to build byte-identical prompts at identical seeds, and
only 25/40 (ARC) and 15/40 (MMLU) chains came back identical: continuous
batching changes batch composition, and therefore the numerics, between runs.
Answers were far more stable than the text that produced them (the flip label
agreed on 79 of those 80 pairs), so treat the seed as reproducing decisions,
not tokens.

Only the standard library is imported here so the analysis package stays
installable without vLLM present; the server itself runs in the notebook.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from mats.backend import Completion


# Slowest per-stream generation rate to plan for on one Kaggle T4. The 09-12
# grid logged 230-270 tok/s aggregate across 16 concurrent requests, about 15
# each, and 14-15 tok/s while a single long chain drained. Planning at 10 keeps
# roughly 40 percent of margin. This is the floor a request is judged against,
# not a measurement of typical speed.
MIN_TOKENS_PER_SECOND = 10.0
# Queueing, prefill and the round trip, none of which scale with max_tokens.
REQUEST_OVERHEAD_SECONDS = 60.0


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
        min_tokens_per_second: float = MIN_TOKENS_PER_SECOND,
        render: Callable[[str, str], str] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/completions"
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.min_tokens_per_second = min_tokens_per_second
        self._render = render or _plain

    def _deadline(self, max_tokens: int) -> float:
        """Seconds to allow a request that may generate `max_tokens` tokens.

        A fixed deadline cannot be right for both ends of this workload: the
        LLM monitor asks for 1,024 tokens and a cue chain for 8,192, and the
        same number is either far too loose for one or fatal for the other.
        """
        return max_tokens / self.min_tokens_per_second + REQUEST_OVERHEAD_SECONDS

    def complete(
        self,
        prompt: str,
        *,
        prefix: str = "",
        seed: int,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        return self.complete_detailed(
            prompt, prefix=prefix, seed=seed, max_tokens=max_tokens, retries=retries
        ).text

    def complete_detailed(
        self,
        prompt: str,
        *,
        prefix: str = "",
        seed: int,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> Completion:
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
        # A single slow or dropped request under heavy concurrency should not
        # kill a multi-hour run - retry transient network/timeout errors with
        # backoff before giving up.
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                deadline = self._deadline(max_tokens)
                with urllib.request.urlopen(request, timeout=deadline) as response:
                    payload = json.loads(response.read())
                choice = payload["choices"][0]
                usage = payload.get("usage") or {}
                return Completion(
                    text=choice["text"],
                    finish_reason=choice.get("finish_reason") or "unknown",
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                )
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                last_error = error
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"vLLM request failed after {retries} attempts") from last_error
