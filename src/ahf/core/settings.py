"""Application settings loaded from environment variables / .env file.

Replaces v1's `config.py` + `sys_config.py` + `app/enums.py` env loading.
Uses Pydantic BaseSettings for type-safe, validated configuration.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ahf.core.enums import DeployEnv, NodeEnv


class Settings(BaseSettings):
    """ahf runtime configuration.

    All fields load from environment variables (or .env file).
    Field names map directly to .env.example keys (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    node_env: NodeEnv = NodeEnv.DEV
    deploy_env: DeployEnv = DeployEnv.DEV
    debug: bool = False
    loglevel: str = "INFO"

    # Bot identity
    bot_id: str = ""
    bot_version: str = "2.0.0-alpha.1"

    # Trading
    exch_mode: str = "SIM"  # SIM | REAL
    pod_dir: str = ""
    env_name: str = "BrunhildEnv-v11"
    agent_id: int = 0
    init_trade_cash: float = 1000.0
    trade_timeout: int = 20
    order_fill_status_timeout: int = 20

    # Pipeline
    pipeline_config: str = "configs/pipeline.json"

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

    # ------------------------------------------------------------------
    # Binance
    # ------------------------------------------------------------------
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_api_key_sim: str = ""
    binance_api_secret_sim: str = ""
    binance_api_key_test: str = ""
    binance_api_secret_test: str = ""

    # ------------------------------------------------------------------
    # LLM keys
    # ------------------------------------------------------------------
    gemini_api_key: str = ""
    openai_api_key: str = ""
    claude_api_key: str = ""

    # ------------------------------------------------------------------
    # Email notifications
    # ------------------------------------------------------------------
    send_mail: bool = False
    send_mail_receiver: str = ""
    google_smtp: str = "smtp.gmail.com"
    google_mail_user: str = ""
    google_mail_secret: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    # ------------------------------------------------------------------
    # Optuna
    # ------------------------------------------------------------------
    optuna_storage: str = "sqlite:///data/optuna.db"
    optuna_study_name: str = "ahf_tune_v2"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("loglevel")
    @classmethod
    def validate_loglevel(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"loglevel must be one of {allowed}, got: {v!r}")
        return upper

    @model_validator(mode="after")
    def validate_pod_dir_for_live(self) -> "Settings":
        """If running in LIVE mode, pod_dir is required."""
        if self.exch_mode.upper() == "REAL" and not self.pod_dir:
            raise ValueError("pod_dir is required when exch_mode=REAL (live trading)")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Use in production code:
        from ahf.core.settings import get_settings
        cfg = get_settings()

    In tests, call get_settings.cache_clear() before each test that
    modifies environment variables.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Convenience alias + extended settings for entrypoints
# ---------------------------------------------------------------------------


class AHFSettings(Settings):
    """Extended settings for entrypoints. Adds fields used by the trade loop."""

    # Trading pair
    symbol: str = "BTCUSDT"

    # Operating mode (LIVE | PAPER | SIMULATION)
    trading_mode: str = "PAPER"

    # Signal pipeline
    min_valid_signals: int = 1
    buy_threshold: float = 0.1
    sell_threshold: float = -0.1
    confidence_floor: float = 0.0

    # Risk management
    max_drawdown_pct: float = 0.15
    max_loss_pct: float = 0.30
    kelly_fraction: float = 0.5
    initial_capital: float = 1000.0

    # Trade loop
    step_interval_seconds: float = 1.0
    halt_on_error: bool = False

    # Audit log
    audit_log_enabled: bool = True
    audit_log_path: str = "data/logs/signal_audit.jsonl"

    # Logging (alias for entrypoints that use log_level)
    @property
    def log_level(self) -> str:
        return self.loglevel

    # Strategy / RL
    strategy: str = "double_kf"
    pod_id: str = "pod_000000"
