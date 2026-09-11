"""
Falcon Sentinel — Health check framework for Falcon subsystems.

Each sentinel is a self-contained check that verifies one subsystem is
functioning correctly. Sentinels return a simple pass/warn/fail result
with a human-readable reason.

Usage:
    from falcon_core.sentinel import SentinelRunner

    runner = SentinelRunner()
    report = runner.run_all()
    for result in report:
        print(f"[{result.status}] {result.name}: {result.reason}")

CLI:
    falcon-sentinel          # run all sentinels
    falcon-sentinel --name database   # run one sentinel
    falcon-sentinel --json   # machine-readable output
"""

from falcon_core.sentinel.base import (
    BaseSentinel,
    SentinelResult,
    SentinelStatus,
    SentinelRunner,
)

__all__ = [
    "BaseSentinel",
    "SentinelResult",
    "SentinelStatus",
    "SentinelRunner",
]
