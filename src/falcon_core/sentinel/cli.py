"""
CLI for falcon-sentinel.

Usage:
    falcon-sentinel                # run all sentinels
    falcon-sentinel --name data-feed   # run one
    falcon-sentinel --json         # machine-readable output
"""

import argparse
import json
import logging
import sys

from falcon_core.sentinel.base import SentinelRunner, SentinelStatus


STATUS_SYMBOLS = {
    SentinelStatus.PASS: "\u2713",  # ✓
    SentinelStatus.WARN: "!",
    SentinelStatus.FAIL: "\u2717",  # ✗
    SentinelStatus.SKIP: "-",
}


def main():
    parser = argparse.ArgumentParser(
        prog="falcon-sentinel",
        description="Run health checks for Falcon subsystems",
    )
    parser.add_argument(
        "--name", "-n",
        help="Run a specific sentinel by name",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_sentinels",
        help="List available sentinels",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    runner = SentinelRunner()

    if args.list_sentinels:
        for name in runner.sentinel_names:
            print(f"  {name}")
        return

    if args.name:
        results = [runner.run_one(args.name)]
    else:
        results = runner.run_all()

    if args.json_output:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print()
        for r in results:
            sym = STATUS_SYMBOLS.get(r.status, "?")
            print(f"  [{sym}] {r.name:.<28s} {r.status.value:>4s}  {r.reason}  ({r.elapsed_ms:.0f}ms)")
        print()

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.status == SentinelStatus.PASS)
        warned = sum(1 for r in results if r.status == SentinelStatus.WARN)
        failed = sum(1 for r in results if r.status == SentinelStatus.FAIL)
        print(f"  {passed}/{total} passed", end="")
        if warned:
            print(f", {warned} warning(s)", end="")
        if failed:
            print(f", {failed} failed", end="")
        print()

    # Exit code: 0 if all pass/warn, 1 if any fail
    if any(r.status == SentinelStatus.FAIL for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
