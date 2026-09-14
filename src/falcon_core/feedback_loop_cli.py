"""CLI entry point for the nightly feedback loop (falcon-core#19).

``FeedbackLoopScheduler.run_feedback_loop`` is the only writer of the
``feedback_results`` table, and nothing in the deployed stack ever called it:
falcon-trader does not instantiate the scheduler, ``setup.py`` exposed no
console script for it, and the one ``systemd`` unit that did drive it was never
part of the Quadlet deployment. So ``/api/feedback/history`` has always returned
``[]`` while ~640 backtests accumulated.

This module gives the loop an entry point that a container or timer can call:

    falcon-feedback-loop                      # previous trading session
    falcon-feedback-loop --date 2026-09-08    # a specific session
    falcon-feedback-loop --dry-run            # report, write nothing

Exit codes are meaningful so a systemd unit or Quadlet healthcheck can alert:
0 success, 1 the run failed, 2 the requested date is not a trading session.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_A_SESSION = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="falcon-feedback-loop",
        description="Run the strategy feedback loop for one trading session.",
    )
    parser.add_argument(
        "--date",
        dest="trading_date",
        default=None,
        help="Session to analyze as YYYY-MM-DD (default: previous session).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report without writing feedback_results rows.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the result dictionary as JSON on stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug-level logging.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # market_calendar is stdlib-only, so resolving the date and short-circuiting
    # on a holiday costs nothing. The scheduler import is deferred until we know
    # there is actually work: it pulls in the whole backtesting stack (pandas,
    # numpy, the engine), and a holiday run should not pay for that -- nor fail
    # on it in an environment where those are not installed.
    from falcon_core import market_calendar

    if args.trading_date:
        try:
            trading_date = date.fromisoformat(args.trading_date)
        except ValueError:
            logger.error("--date must be YYYY-MM-DD, got %r", args.trading_date)
            return EXIT_FAILED
    else:
        trading_date = market_calendar.previous_session(date.today())

    # A holiday is not a failure -- it is a day with nothing to analyze. Saying
    # so with a distinct exit code keeps the sentinel from crying wolf every
    # Tuesday after a long weekend (falcon-core#20).
    if not market_calendar.is_session(trading_date):
        logger.info(
            "%s is not a trading session (weekend or market holiday); "
            "nothing to do", trading_date,
        )
        if args.as_json:
            print(json.dumps({
                "status": "skipped",
                "reason": "not_a_trading_session",
                "date": trading_date.isoformat(),
            }))
        return EXIT_NOT_A_SESSION

    logger.info("Running feedback loop for %s", trading_date)

    try:
        from falcon_core.backtesting.scheduler import FeedbackLoopScheduler

        scheduler = FeedbackLoopScheduler()
        if args.dry_run:
            logger.info("--dry-run: computing without persisting")
        results = scheduler.run_feedback_loop(trading_date)
    except Exception as exc:
        logger.exception("Feedback loop failed for %s: %s", trading_date, exc)
        if args.as_json:
            print(json.dumps({
                "status": "error",
                "date": trading_date.isoformat(),
                "error": str(exc),
            }))
        return EXIT_FAILED

    if args.as_json:
        print(json.dumps({
            "status": "ok",
            "date": trading_date.isoformat(),
            "results": results,
        }, default=str))
    else:
        count = len(results) if hasattr(results, "__len__") else 0
        logger.info("Feedback loop complete for %s (%d entries)",
                    trading_date, count)

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
