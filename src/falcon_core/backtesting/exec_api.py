"""
Backtest EXECUTION API — headless / remote intraday backtest submission.

`results_api.py` already stores and *serves* backtest results. This module adds the missing
*execution* path so a field machine (travel laptop / WSL2 with no `falcon-core` image or flat-file
cache) can trigger an intraday (1-min flat-file) backtest server-side and retrieve results:

    POST /api/backtest      run a backtest (optionally a slippage sweep) and persist it
    GET  /api/backtest/<id> fetch a stored run

Motivation: field request TradingAsBuddies/falcon-strategies#5. Falcon is Quadlet-deployed, so this
capability is a code change + quadlet redeploy — not SSH/runtime access.

Design notes:
- The core, `run_backtest_request(payload, deps)`, is a PURE function with all heavy collaborators
  (data feed, engine, strategy resolver, results store) injected via a `BacktestExecDeps` bundle. It
  imports no Flask / pandas / boto3 at module load, so it unit-tests on a field machine with mocks.
- `default_deps()` wires the REAL collaborators (DataFeed flat files, SimpleBacktestEngine, strategy
  loader, BacktestResultsStore) with lazy imports, and runs only inside the falcon-core image.
- Backtest prices come STRICTLY from Polygon flat files (Massive) — never the REST API / yfinance
  (matches the pipeline mandate). `default_deps()` aborts if flat files are unavailable.
- Slippage is the experiment: `slippage_bps` accepts a list, and one submission runs the whole sweep.
- Schema-loss trap (see falcon-strategies SPEC #2): `BacktestRunRecord` has no mean_R / expectancy /
  track column, so those (plus the per-run slippage) are persisted inside the `parameters` JSON, never
  mapped onto `total_return`.
"""

from __future__ import annotations

import logging

from falcon_core import market_calendar
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Accepted intraday intervals for this endpoint (the flat-file backtest track is 1-min native;
# 5m is allowed for coarser runs). Daily/factor backtests go through their own tools, not here.
_ALLOWED_INTERVALS = {"1m", "5m"}
_DEFAULT_INTERVAL = "1m"
_DEFAULT_COMMISSION = 0.0002       # 2 bps, matches backtest_run.py's realistic liquid-name cost
_DEFAULT_SLIPPAGE_BPS = [3.0]      # 3 bps default when caller omits the sweep
_DEFAULT_CAPITAL = 30000.0


class BacktestRequestError(ValueError):
    """Raised for a malformed backtest request (maps to HTTP 400)."""


@dataclass
class BacktestExecDeps:
    """Injected collaborators. Real wiring is in `default_deps()`; tests pass fakes.

    load_bars(symbol, start, end, interval) -> data object the engine understands (e.g. a DataFrame).
        MUST raise on unavailable flat files rather than silently falling back to REST/yfinance.
    resolve_strategy(name) -> a fresh strategy INSTANCE for each run.
    make_engine(commission, slippage) -> an engine exposing .run(strategy, data, symbol) -> result.
    store_run(record) -> int primary key (or 0). Receives a BacktestRunRecord.
    get_run(run_id) -> Optional[dict] for the GET route.
    """

    load_bars: Callable[[str, str, str, str], Any]
    resolve_strategy: Callable[[str], Any]
    make_engine: Callable[[float, float], Any]
    store_run: Callable[[Any], int]
    get_run: Optional[Callable[[int], Optional[Dict]]] = None


# --------------------------------------------------------------------------- validation

def _as_slippage_list(raw) -> List[float]:
    """Normalise `slippage_bps` into a list of floats. Accepts a scalar or a list."""
    if raw is None:
        return list(_DEFAULT_SLIPPAGE_BPS)
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    out: List[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise BacktestRequestError(f"slippage_bps entry not a number: {v!r}")
        if f < 0:
            raise BacktestRequestError(f"slippage_bps must be >= 0, got {f}")
        out.append(f)
    if not out:
        raise BacktestRequestError("slippage_bps list is empty")
    return out


def _validate(payload: Dict) -> Dict:
    """Validate + normalise the request payload. Raises BacktestRequestError on bad input."""
    if not isinstance(payload, dict):
        raise BacktestRequestError("request body must be a JSON object")

    strategy = payload.get("strategy")
    if not strategy or not isinstance(strategy, str):
        raise BacktestRequestError("'strategy' (string) is required")

    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols or not isinstance(symbols, (list, tuple)) or not all(isinstance(s, str) and s for s in symbols):
        raise BacktestRequestError("'symbols' (non-empty list of tickers) is required")

    start, end = payload.get("start"), payload.get("end")
    if not start or not end:
        raise BacktestRequestError("'start' and 'end' (YYYY-MM-DD) are required")
    parsed = {}
    for label, value in (("start", start), ("end", end)):
        try:
            parsed[label] = date.fromisoformat(value)
        except (TypeError, ValueError):
            raise BacktestRequestError(f"'{label}' must be an ISO date (YYYY-MM-DD), got {value!r}")

    if parsed["start"] > parsed["end"]:
        raise BacktestRequestError(
            f"'start' ({start}) is after 'end' ({end})"
        )

    # A window containing no trading session can never produce bars. Rejecting it
    # here keeps it from arriving at the engine, which used to report the empty
    # load as a 0-trade success (falcon-core#20, #21).
    if not market_calendar.sessions_between(parsed["start"], parsed["end"]):
        raise BacktestRequestError(
            f"window {start}..{end} contains no trading sessions "
            "(weekend or market holiday)"
        )
    if end < start:
        raise BacktestRequestError(f"'end' ({end}) is before 'start' ({start})")

    interval = payload.get("interval", _DEFAULT_INTERVAL)
    if interval not in _ALLOWED_INTERVALS:
        raise BacktestRequestError(
            f"'interval' must be one of {sorted(_ALLOWED_INTERVALS)} for the intraday backtest, got {interval!r}")

    commission = payload.get("commission", _DEFAULT_COMMISSION)
    try:
        commission = float(commission)
    except (TypeError, ValueError):
        raise BacktestRequestError(f"'commission' must be a number, got {commission!r}")

    capital = payload.get("initial_capital", _DEFAULT_CAPITAL)
    try:
        capital = float(capital)
    except (TypeError, ValueError):
        raise BacktestRequestError(f"'initial_capital' must be a number, got {capital!r}")

    return {
        "strategy": strategy,
        "symbols": [s.upper() for s in symbols],
        "start": start,
        "end": end,
        "interval": interval,
        "commission": commission,
        "capital": capital,
        "slippage_bps": _as_slippage_list(payload.get("slippage_bps")),
    }


# --------------------------------------------------------------------------- result shaping

def _summary_from_result(result: Any) -> Dict:
    """Pull a JSON-safe summary from an engine BacktestResult (or its .to_dict())."""
    src = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    keys = ("total_trades", "win_rate", "total_return", "sharpe_ratio",
            "profit_factor", "expectancy", "max_drawdown")
    summary = {k: src.get(k) for k in keys if k in src}
    # Per-trade R distribution is NOT computed by the engine (it needs signal-pairing from
    # backtest_intraday). Expose it only if the result already carries it — never fabricate.
    if "mean_R" in src:
        summary["mean_R"] = src["mean_R"]
    if "trades_R" in src:
        summary["trades_R"] = src["trades_R"]
    return summary


def _build_record(record_cls, *, strategy: str, symbol: str, trading_date: str, interval: str,
                  slippage_bps: float, commission: float, summary: Dict) -> Any:
    """Build a BacktestRunRecord, stashing R/expectancy/track/slippage in `parameters` (schema-loss trap)."""
    return record_cls(
        strategy_name=strategy,
        symbol=symbol,
        trading_date=date.fromisoformat(trading_date),
        interval=interval,
        total_return=float(summary.get("total_return") or 0.0),
        max_drawdown=float(summary.get("max_drawdown") or 0.0),
        sharpe_ratio=float(summary.get("sharpe_ratio") or 0.0),
        win_rate=float(summary.get("win_rate") or 0.0),
        total_trades=int(summary.get("total_trades") or 0),
        signals_count=int(summary.get("total_trades") or 0),
        parameters={
            "track": "intraday",
            "slippage_bps": slippage_bps,
            "commission": commission,
            "expectancy": summary.get("expectancy"),
            "profit_factor": summary.get("profit_factor"),
            "mean_R": summary.get("mean_R"),
        },
    )


# --------------------------------------------------------------------------- core (pure, DI)

def run_backtest_request(payload: Dict, deps: BacktestExecDeps, record_cls=None) -> Dict:
    """Execute a backtest request (optionally a slippage sweep) and persist each run.

    Pure w.r.t. I/O: every side-effecting collaborator is injected via `deps`. Returns a structured
    dict: {strategy, window, runs[], sweep{}, errors[]}. Raises BacktestRequestError on bad input.
    """
    req = _validate(payload)
    if record_cls is None:  # lazy import so the module stays flask/pandas-free at load
        from falcon_core.backtesting.results_api import BacktestRunRecord as record_cls  # type: ignore

    runs: List[Dict] = []
    errors: List[Dict] = []

    for symbol in req["symbols"]:
        try:
            bars = deps.load_bars(symbol, req["start"], req["end"], req["interval"])
        except Exception as e:  # data unavailable for this symbol — record and continue
            errors.append({"symbol": symbol, "stage": "load_bars", "error": str(e)})
            continue
        if bars is None or (hasattr(bars, "__len__") and len(bars) == 0):
            errors.append({"symbol": symbol, "stage": "load_bars", "error": "no bars for window"})
            continue

        for bps in req["slippage_bps"]:
            slippage_frac = bps / 10000.0  # bps -> fraction the engine expects
            try:
                strategy = deps.resolve_strategy(req["strategy"])
                engine = deps.make_engine(req["commission"], slippage_frac)
                result = engine.run(strategy, bars, symbol)
                summary = _summary_from_result(result)
                record = _build_record(
                    record_cls, strategy=req["strategy"], symbol=symbol,
                    trading_date=req["end"], interval=req["interval"],
                    slippage_bps=bps, commission=req["commission"], summary=summary)
                run_id = deps.store_run(record)
            except Exception as e:
                errors.append({"symbol": symbol, "slippage_bps": bps, "stage": "run", "error": str(e)})
                continue
            runs.append({
                "id": run_id,
                "symbol": symbol,
                "slippage_bps": bps,
                "summary": summary,
            })

    return {
        "strategy": req["strategy"],
        "window": {"start": req["start"], "end": req["end"], "interval": req["interval"]},
        "runs": runs,
        "sweep": _sweep_aggregate(runs),
        "errors": errors,
    }


def _sweep_aggregate(runs: List[Dict]) -> List[Dict]:
    """Aggregate runs per slippage level so the edge's sensitivity to slippage is visible at a glance."""
    by_bps: Dict[float, List[Dict]] = {}
    for r in runs:
        by_bps.setdefault(r["slippage_bps"], []).append(r["summary"])
    out = []
    for bps in sorted(by_bps):
        summaries = by_bps[bps]
        def _avg(key):
            vals = [s[key] for s in summaries if s.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None
        out.append({
            "slippage_bps": bps,
            "symbols": len(summaries),
            "avg_total_return": _avg("total_return"),
            "avg_win_rate": _avg("win_rate"),
            "avg_sharpe_ratio": _avg("sharpe_ratio"),
            "avg_expectancy": _avg("expectancy"),
            "total_trades": sum(int(s.get("total_trades") or 0) for s in summaries),
        })
    return out


# --------------------------------------------------------------------------- real deps (falcon-core image)

def default_deps(results_store, *, cache_dir: Optional[str] = None) -> BacktestExecDeps:
    """Wire the REAL collaborators. Runs only inside the falcon-core image (flat files + engine)."""
    import os
    from falcon_core.backtesting.data_feed import DataFeed
    from falcon_core.backtesting.engine import SimpleBacktestEngine

    cache_dir = cache_dir or os.getenv("FALCON_CACHE_DIR", "/var/cache/falcon/flatfiles")
    feed = DataFeed(cache_dir=cache_dir)

    def load_bars(symbol, start, end, interval):
        # Flat files (Massive) ONLY — never REST/yfinance for a backtest.
        if getattr(feed, "flatfiles", None) is None:
            raise RuntimeError(
                "Polygon flat files (Massive) unavailable — backtests must run on the flat files. "
                "Set MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY and run inside the falcon-core image.")
        return feed.get_historical_data(symbol, start, end, interval=interval, source="flatfiles")

    def resolve_strategy(name):
        from falcon_core.backtesting.strategy_loader import load_strategies_from_db
        strategies = {}
        try:
            db = getattr(results_store, "_db_manager", None)
            if db is not None:
                strategies = load_strategies_from_db(db)
        except Exception as e:  # pragma: no cover - env-specific
            logger.warning("load_strategies_from_db failed: %s", e)
        cls = strategies.get(name)
        if cls is None:
            raise BacktestRequestError(
                f"unknown strategy {name!r}; available: {sorted(strategies) or 'none loaded'}")
        return cls()

    def make_engine(commission, slippage):
        return SimpleBacktestEngine(initial_capital=_DEFAULT_CAPITAL, commission=commission, slippage=slippage)

    def store_run(record):
        return results_store.store_backtest_run(record)

    def get_run(run_id):
        rows = results_store.get_recent_backtests(days=3650, limit=100000)
        for row in rows or []:
            if int(row.get("id", -1)) == int(run_id):
                return row
        return None

    return BacktestExecDeps(
        load_bars=load_bars, resolve_strategy=resolve_strategy, make_engine=make_engine,
        store_run=store_run, get_run=get_run)


# --------------------------------------------------------------------------- Flask routes

def create_backtest_exec_routes(app, results_store, deps: Optional[BacktestExecDeps] = None):
    """Register the execution routes on the Flask `app`. Mirrors results_api.create_api_routes."""
    from flask import jsonify, request

    if deps is None:
        deps = default_deps(results_store)

    @app.route('/api/backtest', methods=['POST'])
    def submit_backtest():
        try:
            payload = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "request body must be valid JSON"}), 400
        try:
            result = run_backtest_request(payload, deps)
        except BacktestRequestError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("backtest execution failed")
            return jsonify({"error": f"backtest execution failed: {e}"}), 500
        status = 200 if result["runs"] else 422  # nothing ran (all symbols failed) -> 422
        return jsonify(result), status

    @app.route('/api/backtest/<int:run_id>', methods=['GET'])
    def fetch_backtest(run_id):
        getter = deps.get_run
        if getter is None:
            return jsonify({"error": "run lookup not available"}), 501
        row = getter(run_id)
        if row is None:
            return jsonify({"error": f"backtest run {run_id} not found"}), 404
        return jsonify(row)

    logger.info("Registered backtest execution routes (POST /api/backtest, GET /api/backtest/<id>)")
