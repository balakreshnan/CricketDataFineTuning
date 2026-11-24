"""NeMo Gym resources server: T20 cricket questions with exact numeric answers.

One task = one question from the held-out split of Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full. The model answers
once (single turn); verify() parses "Final answer: <number>" and compares it with the exact answer using the same
rules as the rest of this project (cricket_rules.py is a copy of rhea/cricket_qa.py). Reward 1.0 if the answer is
finished and correct, else 0.0.
"""
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_rules as Q  # noqa: E402


class CricketQAResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class CricketQAVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    qid: str
    qtype: str
    answer_kind: str
    gold: float


class CricketQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    correct: bool
    finished: bool
    pred: Optional[Any] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class CricketQAVerifier:
    async def verify(self, body: CricketQAVerifyRequest) -> CricketQAVerifyResponse:
        resp = body.response
        text = resp.output_text or ""
        finished = getattr(resp, "status", "completed") in (None, "completed") and getattr(resp, "incomplete_details", None) is None
        # vLLM's reasoning parser has already removed the thinking part, so output_text is the answer part only. The
        # project's parser expects the raw text with its closing think marker; an empty answer part means the model
        # never finished thinking, which counts as no answer, exactly as in the other benchmarks of this project.
        pred = Q.parse_answer("</think>" + text) if text.strip() else None
        correct = bool(finished and Q.is_correct(pred, body.gold, body.answer_kind))
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "output_tokens_details", None)
        return CricketQAVerifyResponse(**body.model_dump(), reward=float(correct), correct=correct, finished=finished, pred=pred,
                                       output_tokens=getattr(usage, "output_tokens", None),
                                       reasoning_tokens=getattr(details, "reasoning_tokens", None))


class CricketQAResourcesServer(CricketQAVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: CricketQAResourcesServerConfig


if __name__ == "__main__":
    CricketQAResourcesServer.run_webserver()
