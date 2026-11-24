"""NeMo Gym resources server for NeMo RL training: T20 cricket questions with exact numeric answers.

Same task and checker as rhea/nemotron/gym/cricket_qa/app.py (used for evaluation with nemo-gym 0.7.0), written for
the NeMo Gym version pinned by NeMo RL (classes live in nemo_gym.server_runtime there), with one addition: the reward
can carry the length term of rhea/rlvr.py, so NeMo RL optimises the same objective:

    reward = 1 - len_coef * output_tokens / max_output_tokens   if the answer is finished and correct, else 0

len_coef = 0 gives the plain 0/1 reward used for evaluation.
"""
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_rules as Q  # noqa: E402


class CricketQAResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    len_coef: float = 0.0
    max_output_tokens: int = 4096


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
    evaluation_completed: bool = True


class CricketQAResourcesServer(SimpleResourcesServer):
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    config: CricketQAResourcesServerConfig

    async def verify(self, body: CricketQAVerifyRequest) -> CricketQAVerifyResponse:
        resp = body.response
        text = resp.output_text or ""
        finished = getattr(resp, "status", "completed") in (None, "completed") and getattr(resp, "incomplete_details", None) is None
        # The reasoning parser has already removed the thinking part; the project parser expects the closing marker.
        pred = Q.parse_answer("</think>" + text) if text.strip() else None
        correct = bool(finished and Q.is_correct(pred, body.gold, body.answer_kind))
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "output_tokens_details", None)
        out_tokens = getattr(usage, "output_tokens", None)
        reward = 0.0
        if correct:
            frac = min(out_tokens or 0, self.config.max_output_tokens) / self.config.max_output_tokens
            reward = 1.0 - self.config.len_coef * frac
        return CricketQAVerifyResponse(**body.model_dump(), reward=reward, correct=correct, finished=finished, pred=pred,
                                       output_tokens=out_tokens,
                                       reasoning_tokens=getattr(details, "reasoning_tokens", None))


if __name__ == "__main__":
    CricketQAResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CricketQAResourcesServer.run_webserver()  # noqa: F401
