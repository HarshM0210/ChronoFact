from dataclasses import dataclass
from datetime import datetime


@dataclass
class Turn:
    user_id: str
    session_id: str
    turn_idx: int
    speaker: str            # "user" | "assistant"
    text: str
    ts: datetime


@dataclass
class CallStats:            # summed over the LLM calls of one operation
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    def add(self, o: "CallStats") -> "CallStats":
        self.llm_calls += o.llm_calls
        self.tokens_in += o.tokens_in
        self.tokens_out += o.tokens_out
        self.latency_ms += o.latency_ms
        return self
