"""LLMSignalProducer — uses an LLM API to generate a trading signal.

The LLM is given market data context and asked to reason about the
optimal trade direction. Output is parsed from structured JSON.

Falls back to HOLD with confidence=0.0 if the LLM is unavailable or
returns unparseable output (never blocks the pipeline).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """You are a crypto trading signal generator.

Market data (current step):
{market_context}

Based on this data, provide a trading signal as JSON:
{{"action": <float in [-1.0, 1.0]>, "confidence": <float in [0.0, 1.0]>, "reasoning": "<brief>"}}

Rules:
- action: -1.0=strong sell, 0.0=hold, +1.0=strong buy
- confidence: your certainty in the signal
- Output ONLY the JSON object.
"""


class LLMSignalProducer(SignalProducer):
    """LLM-based signal source with structured JSON output parsing."""

    def __init__(
        self,
        name: str,
        client: Any = None,
        prompt_template: str = _DEFAULT_PROMPT,
        model: str = "",
    ) -> None:
        self._name = name
        self._client = client
        self._prompt_template = prompt_template
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        if self._client is None:
            return SignalOutput(signal_name=self.name, action=0.0, confidence=0.0)

        try:
            market_context = self._format_market_context(market_data)
            prompt = self._prompt_template.format(market_context=market_context)
            response = (
                self._client.generate(prompt, model=self._model)
                if self._model
                else self._client.generate(prompt)
            )
            parsed = json.loads(response.strip())
            return SignalOutput(
                signal_name=self.name,
                action=float(parsed["action"]),
                confidence=float(parsed["confidence"]),
                metadata={"reasoning": parsed.get("reasoning", ""), "model": self._model},
            )
        except Exception as e:
            logger.warning(
                "LLMSignalProducer failed — returning HOLD",
                extra={"producer": self.name, "error": str(e)},
            )
            return SignalOutput(signal_name=self.name, action=0.0, confidence=0.0)

    def health_check(self) -> None:
        if self._client is None:
            raise RuntimeError(f"[{self.name}] LLM client not configured")

    def _format_market_context(self, market_data: dict) -> str:
        """Extract key metrics from market_data for the prompt."""
        lines = []
        for tf, data in market_data.items():
            if isinstance(data, dict) and "last_close" in data:
                lines.append(f"  {tf}: close={data['last_close']}, volume={data.get('last_volume', 'N/A')}")
        return "\n".join(lines) if lines else "No market data available"

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "LLMSignalProducer":
        model = config.get("model", "")
        client_key = config.get("client_key", model)
        client = runtime_deps.get(client_key) or runtime_deps.get("default_llm_client")
        return cls(
            name=name,
            client=client,
            prompt_template=config.get("prompt_template", _DEFAULT_PROMPT),
            model=model,
        )
