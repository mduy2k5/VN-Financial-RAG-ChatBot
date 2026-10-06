"""Explicit token budgets, with a conservative estimate when no tokenizer is available."""
from dataclasses import dataclass
from typing import Callable


def estimate_tokens(text: str) -> int:
    # UTF-8 bytes deliberately overestimate ordinary byte-BPE text. It is an
    # estimate, not a claim that every model shares a tokenizer.
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class TokenBudget:
    context_window: int
    output_reserve: int
    instruction_tokens: int = 0
    history_tokens: int = 0
    query_tokens: int = 0
    margin: int = 128

    @property
    def available(self) -> int:
        return max(0, self.context_window - self.output_reserve - self.instruction_tokens
                   - self.history_tokens - self.query_tokens - self.margin)

    def pack(self, blocks: list[dict], counter: Callable[[str], int] = estimate_tokens) -> tuple:
        selected: list[dict] = []
        omitted: list[dict] = []
        used = 0
        for block in blocks:
            tokens = counter(block["text"]) + (counter("\n---\n") if selected else 0)
            if used + tokens > self.available:
                omitted.append({"id": block["id"], "reason": "token_budget", "tokens": tokens})
                continue
            selected.append(block)
            used += tokens
        return selected, {"budget_tokens": self.available, "estimated_tokens": used,
                          "estimator": "utf8_bytes_upper_estimate" if counter is estimate_tokens
                          else "model_tokenizer", "omitted": omitted}
