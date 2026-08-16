"""MetaLLMAggregator — LLM-as-judge meta-reasoner.

Presents all producer signals to an LLM and asks it to reason about
the overall market direction and produce a final consensus signal.

Falls back to WeightedVoteAggregator silently if:
- No LLM client is configured
- The LLM API call fails
- The response cannot be parsed as valid JSON

This ensures MetaLLM never blocks a trade step.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ahf.signals.aggregators.weighted_vote import WeightedVoteAggregator
from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)

_FALLBACK = WeightedVoteAggregator()

_DEFAULT_PROMPT_TEMPLATE = """You are a systematic trading signal aggregator.

Below are trading signals from multiple independent models for the same asset:

{signals_json}

Each signal has:
- signal_name: the model that produced it
- action: a number in [-1.0, 1.0] where -1=strong sell, 0=hold, +1=strong buy
- confidence: how confident the model is in [0.0, 1.0]

Your task:
1. Reason briefly about the signals and any disagreements.
2. Output a final consensus signal as JSON with exactly these keys:
   {{"action": <float in [-1.0, 1.0]>, "confidence": <float in [0.0, 1.0]>, "reasoning": "<one sentence>"}}

Output ONLY the JSON object, nothing else.
"""


class MetaLLMAggregator(SignalAggregator):
    """LLM-as-meta-reasoner aggregator with WeightedVote fallback.

    Config options:
        client_key (str): Key in runtime_deps for the LLM client object.
            The client must have a .generate(prompt: str) -> str method.
        prompt_template (str, optional): Custom prompt template.
            Must contain {signals_json} placeholder.
        model (str, optional): Model identifier passed to client.generate().
    """

    def __init__(
        self,
        client: Any = None,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
        model: str = "",
    ) -> None:
        self._client = client
        self._prompt_template = prompt_template
        self._model = model

    def aggregate(
        self,
        signals: list[Optional[SignalOutput]],
        context: dict,
    ) -> SignalOutput:
        valid = [s for s in signals if s is not None]

        if not valid or self._client is None:
            return _FALLBACK.aggregate(signals, context)

        try:
            signals_data = [s.to_audit_dict() for s in valid]
            prompt = self._prompt_template.format(
                signals_json=json.dumps(signals_data, indent=2)
            )
            response_text = self._client.generate(prompt, model=self._model) if self._model else self._client.generate(prompt)
            parsed = json.loads(response_text.strip())

            return SignalOutput(
                signal_name="meta_llm",
                action=float(parsed["action"]),
                confidence=float(parsed["confidence"]),
                metadata={"reasoning": parsed.get("reasoning", "")},
            )
        except Exception as e:
            logger.warning(
                "MetaLLMAggregator falling back to WeightedVote",
                extra={"error": str(e)},
            )
            fallback = _FALLBACK.aggregate(signals, context)
            return SignalOutput(
                signal_name="meta_llm_fallback",
                action=fallback.action,
                confidence=fallback.confidence,
                metadata={"fallback_reason": str(e)},
            )

    def health_check(self) -> None:
        if self._client is None:
            logger.warning("MetaLLMAggregator: no LLM client configured — will use WeightedVote fallback")

    @classmethod
    def from_config(cls, config: dict, runtime_deps: dict) -> "MetaLLMAggregator":
        client_key = config.get("client_key", config.get("model", "default_llm_client"))
        client = runtime_deps.get(client_key) or runtime_deps.get("default_llm_client")
        return cls(
            client=client,
            prompt_template=config.get("prompt_template", _DEFAULT_PROMPT_TEMPLATE),
            model=config.get("model", ""),
        )
