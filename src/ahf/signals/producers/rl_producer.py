"""RLSignalProducer — wraps BrunhildEnv_v11 / DRLAgent for live inference.

In live/paper trading, this producer:
1. Calls the DRLAgent to get the next action given the latest market observation.
2. Normalises the raw action (int -1/0/1) to a float in [-1.0, 1.0].
3. Uses the agent's confidence proxy (e.g. policy entropy or fixed value).

The RL env is expected to be pre-built by the caller and passed via runtime_deps.
This keeps the producer stateless with respect to env setup.

Phase 4 implementation — stub fallback if env not provided.
"""
from __future__ import annotations

import logging
from typing import Any

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)


class RLSignalProducer(SignalProducer):
    """RL agent inference producer.

    Wraps a DRLAgent / AgentPPO instance. The agent's predict() or
    get_action() method is called with the current obs from the env.

    Args:
        name: Producer identifier.
        agent: A pre-loaded AgentPPO / DRLAgent instance.
            Must have: agent.act_no_exploration(obs) -> int
        env: The BrunhildEnv_v11 instance for observation generation.
            Must have: env.get_obs() -> np.ndarray
        confidence: Fixed confidence value to use (0.0–1.0).
            Phase 4 uses a fixed value; Phase 5 will use policy entropy.
        model_path: Path to the model checkpoint (for audit logging).
        agent_type: Agent architecture name (e.g. 'ppo').
    """

    def __init__(
        self,
        name: str,
        agent: Any = None,
        env: Any = None,
        confidence: float = 0.7,
        model_path: str = "",
        agent_type: str = "ppo",
    ) -> None:
        self._name = name
        self._agent = agent
        self._env = env
        self._confidence = confidence
        self._model_path = model_path
        self._agent_type = agent_type

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """Run RL agent inference.

        If agent or env is not set, returns a neutral HOLD signal.
        """
        if self._agent is None or self._env is None:
            logger.debug(
                "RLSignalProducer: agent/env not set — returning HOLD stub",
                extra={"producer": self._name},
            )
            return SignalOutput(signal_name=self._name, action=0.0, confidence=0.5)

        try:
            # Get observation from the env
            obs = self._env.get_obs(market_data)

            # Run agent inference — returns discrete action (-1, 0, 1)
            raw_action = int(self._agent.act_no_exploration(obs))

            # Normalise: -1 → -1.0 (sell), 0 → 0.0 (hold), 1 → 1.0 (buy)
            action = float(max(-1, min(1, raw_action)))

            return SignalOutput(
                signal_name=self._name,
                action=action,
                confidence=self._confidence,
                metadata={
                    "agent_type": self._agent_type,
                    "model_path": self._model_path,
                    "raw_action": raw_action,
                },
            )

        except Exception as e:
            logger.warning(
                "RLSignalProducer inference failed — returning HOLD",
                extra={"producer": self._name, "error": str(e)},
            )
            return SignalOutput(signal_name=self._name, action=0.0, confidence=0.0)

    def health_check(self) -> None:
        if self._agent is None:
            raise RuntimeError(f"[{self._name}] RL agent not loaded — call from_config with valid runtime_deps")
        if self._env is None:
            raise RuntimeError(f"[{self._name}] RL env not set — pass env in runtime_deps")

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "RLSignalProducer":
        """Build from pipeline.json config.

        Expected runtime_deps keys:
            "{name}_agent" or "rl_agent": The pre-loaded agent instance.
            "{name}_env" or "rl_env":     The pre-built RL environment.

        Example pipeline.json config:
            {
                "id": "rl_ppo",
                "type": "rl",
                "config": {
                    "model_path": "data/models/pod_000000/PROD/...",
                    "agent": "ppo",
                    "confidence": 0.7
                }
            }
        """
        agent = (
            runtime_deps.get(f"{name}_agent")
            or runtime_deps.get("rl_agent")
        )
        env = (
            runtime_deps.get(f"{name}_env")
            or runtime_deps.get("rl_env")
        )
        return cls(
            name=name,
            agent=agent,
            env=env,
            confidence=float(config.get("confidence", 0.7)),
            model_path=config.get("model_path", ""),
            agent_type=config.get("agent", "ppo"),
        )
