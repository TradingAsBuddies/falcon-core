"""
Sentinel base classes.

Designed to be decoupled from Falcon internals — sentinels receive their
dependencies via constructor injection so the framework can be lifted
into a standalone package later.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SentinelStatus(str, Enum):
    """Outcome of a sentinel check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class SentinelResult:
    """Outcome of a single sentinel run."""
    name: str
    status: SentinelStatus
    reason: str
    elapsed_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "details": self.details,
        }


class BaseSentinel(ABC):
    """
    A single health check.

    Subclasses implement ``check()`` and return a ``SentinelResult``.
    The runner calls ``run()`` which handles timing and error capture.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short kebab-case identifier (e.g. ``data-feed``)."""

    @property
    def description(self) -> str:
        """One-line human description (optional)."""
        return ""

    @abstractmethod
    def check(self) -> SentinelResult:
        """Execute the health check. Must not raise."""

    def run(self) -> SentinelResult:
        """Run the sentinel with timing and error capture."""
        t0 = time.monotonic()
        try:
            result = self.check()
        except Exception as e:
            result = SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Unhandled exception: {e}",
            )
            logger.exception("Sentinel %s raised", self.name)
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result


class SentinelRunner:
    """
    Discovers and runs sentinels.

    By default loads all built-in sentinels. Pass ``sentinels=`` to
    override with a custom list (useful for testing or Colosseum).
    """

    def __init__(self, sentinels: Optional[List[BaseSentinel]] = None):
        if sentinels is not None:
            self._sentinels = sentinels
        else:
            self._sentinels = self._load_builtins()

    def _load_builtins(self) -> List[BaseSentinel]:
        """Import and instantiate all built-in sentinels."""
        loaded = []

        # Each import is wrapped so one broken sentinel doesn't block the rest.
        sentinel_factories = [
            ("falcon_core.sentinel.check_database", "DatabaseSentinel"),
            ("falcon_core.sentinel.check_data_feed", "DataFeedSentinel"),
            ("falcon_core.sentinel.check_data_feed", "DataFeedMinuteSentinel"),
            ("falcon_core.sentinel.check_data_feed", "PolygonMinuteSentinel"),
            ("falcon_core.sentinel.check_strategy_roster", "StrategyRosterSentinel"),
            ("falcon_core.sentinel.check_backtest_engine", "BacktestEngineSentinel"),
            ("falcon_core.sentinel.check_timezone", "TimezoneSentinel"),
            ("falcon_core.sentinel.check_market_page", "MarketPageSentinel"),
        ]

        for module_path, class_name in sentinel_factories:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                loaded.append(cls())
            except Exception as e:
                logger.warning("Could not load sentinel %s.%s: %s", module_path, class_name, e)

        return loaded

    @property
    def sentinel_names(self) -> List[str]:
        return [s.name for s in self._sentinels]

    def run_all(self) -> List[SentinelResult]:
        """Run every sentinel and return results."""
        results = []
        for sentinel in self._sentinels:
            logger.info("Running sentinel: %s", sentinel.name)
            results.append(sentinel.run())
        return results

    def run_one(self, name: str) -> SentinelResult:
        """Run a single sentinel by name."""
        for sentinel in self._sentinels:
            if sentinel.name == name:
                return sentinel.run()
        return SentinelResult(
            name=name,
            status=SentinelStatus.SKIP,
            reason=f"No sentinel named '{name}'. Available: {self.sentinel_names}",
        )
