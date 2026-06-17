#!/usr/bin/env python3
"""
AI Strategy Advisor

Analyzes strategy backtest results and proposes improvements using Claude AI.
Includes budget enforcement and cost tracking per strategy.

Two classes:
  - CostTracker: Budget management and API usage logging
  - StrategyAdvisor: AI analysis and proposal generation
"""

import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cost per 1M tokens by model (USD). Bare current model ids matching the rest
# of the repo (falcon-strategies/tools/codify.py uses 'claude-opus-4-8').
MODEL_COSTS = {
    'claude-opus-4-8': {'input': 5.00, 'output': 25.00},
    'claude-haiku-4-5': {'input': 1.00, 'output': 5.00},
}

DEFAULT_MODEL = 'claude-haiku-4-5'
DEFAULT_MAX_NO_IMPROVEMENT = 5


class CostTracker:
    """Budget enforcement and API usage logging for the advisor system."""

    def __init__(self, db):
        self.db = db

    def get_budget(self, strategy_name: str) -> Dict[str, Any]:
        """Get or create budget record for a strategy."""
        row = self.db.execute(
            'SELECT * FROM strategy_advisor_budget WHERE strategy_name = %s',
            (strategy_name,), fetch='one'
        )

        if row:
            if isinstance(row, dict):
                return row
            # SQLite Row — convert to dict by column names
            return dict(row)

        # Create new budget record
        now = datetime.now().isoformat()
        self.db.execute(
            '''INSERT INTO strategy_advisor_budget
               (strategy_name, created_at, updated_at)
               VALUES (%s, %s, %s)''',
            (strategy_name, now, now)
        )

        return {
            'strategy_name': strategy_name,
            'monthly_budget_usd': 1.00,
            'total_spent_usd': 0.0,
            'current_month_spent_usd': 0.0,
            'months_active': 0,
            'max_months': 4,
            'consecutive_no_improvement': 0,
            'last_improvement_at': None,
            'status': 'active',
        }

    def can_spend(self, strategy_name: str, estimated_cost: float) -> Tuple[bool, str]:
        """Check if a strategy has budget for the estimated cost.

        Returns:
            (can_spend, reason) — reason is empty if allowed
        """
        budget = self.get_budget(strategy_name)

        status = budget.get('status', 'active')
        if status != 'active':
            return False, f"Strategy budget status is '{status}'"

        monthly_budget = float(budget.get('monthly_budget_usd', 1.00))
        month_spent = float(budget.get('current_month_spent_usd', 0.0))

        if month_spent + estimated_cost > monthly_budget:
            return False, (
                f"Would exceed monthly budget: "
                f"${month_spent:.4f} + ${estimated_cost:.4f} > ${monthly_budget:.2f}"
            )

        no_improvement = int(budget.get('consecutive_no_improvement', 0))
        if no_improvement >= DEFAULT_MAX_NO_IMPROVEMENT:
            return False, (
                f"Consecutive no-improvement limit reached ({no_improvement})"
            )

        max_months = int(budget.get('max_months', 4))
        months_active = int(budget.get('months_active', 0))
        if months_active > max_months:
            return False, f"Max months exceeded ({months_active} > {max_months})"

        return True, ""

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of an API call."""
        costs = MODEL_COSTS.get(model, MODEL_COSTS[DEFAULT_MODEL])
        cost = (
            (input_tokens / 1_000_000) * costs['input']
            + (output_tokens / 1_000_000) * costs['output']
        )
        return round(cost, 6)

    def record_usage(
        self,
        strategy_name: Optional[str],
        service: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str,
    ) -> float:
        """Log API usage and update budget counters.

        Returns:
            Actual cost in USD
        """
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        now = datetime.now().isoformat()

        # Log to api_usage
        self.db.execute(
            '''INSERT INTO api_usage
               (service, model, strategy_name, input_tokens, output_tokens,
                cost_usd, request_type, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
            (service, model, strategy_name, input_tokens, output_tokens,
             cost, request_type, now)
        )

        # Update budget if strategy-specific
        if strategy_name:
            self.db.execute(
                '''UPDATE strategy_advisor_budget SET
                   total_spent_usd = total_spent_usd + %s,
                   current_month_spent_usd = current_month_spent_usd + %s,
                   updated_at = %s
                   WHERE strategy_name = %s''',
                (cost, cost, now, strategy_name)
            )

        return cost

    def record_improvement(self, strategy_name: str, improved: bool):
        """Track improvement history for budget decisions."""
        now = datetime.now().isoformat()

        if improved:
            self.db.execute(
                '''UPDATE strategy_advisor_budget SET
                   consecutive_no_improvement = 0,
                   last_improvement_at = %s,
                   updated_at = %s
                   WHERE strategy_name = %s''',
                (now, now, strategy_name)
            )
        else:
            self.db.execute(
                '''UPDATE strategy_advisor_budget SET
                   consecutive_no_improvement = consecutive_no_improvement + 1,
                   updated_at = %s
                   WHERE strategy_name = %s''',
                (now, strategy_name)
            )

            # Check if should auto-retire
            budget = self.get_budget(strategy_name)
            no_imp = int(budget.get('consecutive_no_improvement', 0))
            if no_imp >= DEFAULT_MAX_NO_IMPROVEMENT:
                self.db.execute(
                    '''UPDATE strategy_advisor_budget SET
                       status = %s, updated_at = %s
                       WHERE strategy_name = %s''',
                    ('retired', now, strategy_name)
                )
                logger.info(
                    f"Auto-retired advisor budget for '{strategy_name}' "
                    f"after {no_imp} consecutive no-improvement cycles"
                )

    def reset_monthly_budgets(self):
        """Reset monthly counters for all active strategies. Call at month start."""
        now = datetime.now().isoformat()
        self.db.execute(
            '''UPDATE strategy_advisor_budget SET
               current_month_spent_usd = 0,
               months_active = months_active + 1,
               updated_at = %s
               WHERE status = %s''',
            (now, 'active')
        )
        logger.info("Monthly budgets reset for all active strategies")

    def get_all_budgets(self) -> List[Dict]:
        """Get budget status for all strategies."""
        rows = self.db.execute(
            'SELECT * FROM strategy_advisor_budget ORDER BY strategy_name',
            fetch='all'
        )
        if not rows:
            return []
        return [dict(r) if hasattr(r, 'keys') else r for r in rows]


class StrategyAdvisor:
    """AI-powered strategy analysis and proposal generation."""

    def __init__(self, db, model: str = None):
        self.db = db
        self.model = model or os.getenv('FALCON_ADVISOR_MODEL', DEFAULT_MODEL)
        self.cost_tracker = CostTracker(db)
        self._client = None

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. "
                    "Install with: pip install falcon-core[advisor]"
                )
            api_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "No API key set: provide CLAUDE_API_KEY or ANTHROPIC_API_KEY"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _get_historical_proposals(self, strategy_name: str, limit: int = 5) -> List[Dict]:
        """Get recent proposals for context (avoid repeating failed ideas)."""
        rows = self.db.execute(
            '''SELECT proposal_type, change_description, status,
                      proposed_sharpe, proposed_win_rate, proposed_total_return
               FROM strategy_proposals
               WHERE strategy_name = %s
               ORDER BY created_at DESC LIMIT %s''',
            (strategy_name, limit), fetch='all'
        )
        if not rows:
            return []
        return [dict(r) if hasattr(r, 'keys') else r for r in rows]

    def analyze_and_propose(
        self,
        strategy_name: str,
        strategy_code: str,
        backtest_results: Dict[str, Any],
        historical_proposals: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        """Analyze strategy performance and propose an improvement.

        Args:
            strategy_name: Name of the strategy
            strategy_code: Current source code
            backtest_results: Dict with metrics (sharpe, win_rate, total_return, etc.)
            historical_proposals: Previous proposals for context

        Returns:
            Proposal dict, or None if budget exceeded or error
        """
        # Pre-flight budget check (~$0.01 per Haiku call)
        estimated_cost = self.cost_tracker.estimate_cost(self.model, 4000, 2000)
        can_spend, reason = self.cost_tracker.can_spend(strategy_name, estimated_cost)
        if not can_spend:
            logger.info(f"Skipping '{strategy_name}': {reason}")
            return None

        # Data-sufficiency gate (runs AFTER the dollar pre-check, BEFORE any
        # Claude call so tiny samples spend zero tokens). Require >= 30 trades
        # across the suite — fewer is not enough signal to propose a non-overfit
        # change per the Carver out-of-sample discipline.
        total_trades = backtest_results.get('n_trades')
        if total_trades is None:
            per_symbol = backtest_results.get('per_symbol') or []
            summed = 0
            have_any = False
            for ps in per_symbol:
                nt = ps.get('n_trades') if isinstance(ps, dict) else None
                if nt is not None:
                    summed += int(nt)
                    have_any = True
            total_trades = summed if have_any else None
        if total_trades is None or int(total_trades) < 30:
            logger.info(
                f"Skipping '{strategy_name}': insufficient backtest data "
                f"(total_trades={total_trades} < 30); no Claude call made"
            )
            return None

        if historical_proposals is None:
            historical_proposals = self._get_historical_proposals(strategy_name)

        # Build the prompt
        prompt = self._build_prompt(
            strategy_name, strategy_code, backtest_results, historical_proposals
        )

        # Call Claude
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.3,
                system=(
                    "You are an expert quantitative trading strategy developer. "
                    "You analyze trading strategy code and backtest results, "
                    "then propose exactly ONE small, targeted improvement. "
                    "Always respond with valid JSON."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error(f"Claude API call failed for '{strategy_name}': {e}")
            return None

        # Record usage
        usage = response.usage
        cost = self.cost_tracker.record_usage(
            strategy_name=strategy_name,
            service='advisor',
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            request_type='analyze_and_propose',
        )

        # Parse response
        content = response.content[0].text
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            logger.error(f"Could not parse JSON from advisor response for '{strategy_name}'")
            return None

        try:
            proposal_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in advisor response for '{strategy_name}': {e}")
            return None

        proposed_code = proposal_data.get('proposed_code', '')
        if not proposed_code:
            logger.warning(f"No proposed_code in advisor response for '{strategy_name}'")
            return None

        # Validate proposed code
        from falcon_core.backtesting.strategy_loader import validate_strategy_code
        is_valid, error = validate_strategy_code(proposed_code)
        if not is_valid:
            logger.warning(
                f"Proposed code for '{strategy_name}' failed validation: {error}"
            )
            return None

        # Store proposal
        now = datetime.now().isoformat()
        proposal_id = self.db.execute(
            '''INSERT INTO strategy_proposals
               (strategy_name, proposal_type, current_code, proposed_code,
                analysis_summary, change_description, expected_improvement,
                api_cost_usd, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
            (
                strategy_name,
                proposal_data.get('proposal_type', 'code_change'),
                strategy_code,
                proposed_code,
                proposal_data.get('analysis_summary', ''),
                proposal_data.get('change_description', ''),
                proposal_data.get('expected_improvement', ''),
                cost,
                now,
            )
        )

        return {
            'id': proposal_id,
            'strategy_name': strategy_name,
            'proposal_type': proposal_data.get('proposal_type', 'code_change'),
            'analysis_summary': proposal_data.get('analysis_summary', ''),
            'change_description': proposal_data.get('change_description', ''),
            'expected_improvement': proposal_data.get('expected_improvement', ''),
            'api_cost_usd': cost,
        }

    def backtest_proposal(self, proposal_id: int) -> Optional[Dict]:
        """Run backtest comparison between current and proposed code.

        Args:
            proposal_id: ID from strategy_proposals table

        Returns:
            Comparison dict with metrics, or None on error
        """
        from falcon_core.backtesting.strategy_loader import load_strategy_from_code

        row = self.db.execute(
            '''SELECT strategy_name, current_code, proposed_code
               FROM strategy_proposals WHERE id = %s''',
            (proposal_id,), fetch='one'
        )
        if not row:
            logger.error(f"Proposal {proposal_id} not found")
            return None

        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        current_code = row['current_code'] if isinstance(row, dict) else row[1]
        proposed_code = row['proposed_code'] if isinstance(row, dict) else row[2]

        # Load both strategy versions
        current_cls = load_strategy_from_code(current_code, f"{name}_current")
        proposed_cls = load_strategy_from_code(proposed_code, f"{name}_proposed")

        if not current_cls or not proposed_cls:
            logger.error(f"Could not load strategy classes for proposal {proposal_id}")
            return None

        # Get symbols from roster
        roster_row = self.db.execute(
            'SELECT symbols, interval FROM strategy_roster WHERE strategy_name = %s',
            (name,), fetch='one'
        )
        if not roster_row:
            logger.error(f"Strategy '{name}' not in roster")
            return None

        symbols_raw = roster_row['symbols'] if isinstance(roster_row, dict) else roster_row[0]
        interval = roster_row['interval'] if isinstance(roster_row, dict) else roster_row[1]
        symbols = json.loads(symbols_raw) if isinstance(symbols_raw, str) else (symbols_raw or [])

        # Run backtests
        try:
            from falcon_core.backtesting.data_feed import DataFeed
            from falcon_core.backtesting.engine import SimpleBacktestEngine
            from datetime import timedelta, date as date_type
        except ImportError as e:
            logger.error(f"Backtesting dependencies not available: {e}")
            return None

        feed = DataFeed(db_manager=self.db)
        end_date = date_type.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=42)

        current_metrics = self._run_backtest_suite(
            current_cls, feed, symbols, interval, start_date, end_date
        )
        proposed_metrics = self._run_backtest_suite(
            proposed_cls, feed, symbols, interval, start_date, end_date
        )

        # Update proposal with comparison
        now = datetime.now().isoformat()
        self.db.execute(
            '''UPDATE strategy_proposals SET
               current_sharpe = %s, proposed_sharpe = %s,
               current_win_rate = %s, proposed_win_rate = %s,
               current_total_return = %s, proposed_total_return = %s
               WHERE id = %s''',
            (
                current_metrics.get('sharpe'),
                proposed_metrics.get('sharpe'),
                current_metrics.get('win_rate'),
                proposed_metrics.get('win_rate'),
                current_metrics.get('total_return'),
                proposed_metrics.get('total_return'),
                proposal_id,
            )
        )

        return {
            'proposal_id': proposal_id,
            'current': current_metrics,
            'proposed': proposed_metrics,
        }

    def _run_backtest_suite(self, strategy_cls, feed, symbols, interval, start_date, end_date):
        """Run backtests across all symbols and aggregate metrics."""
        from falcon_core.backtesting.engine import SimpleBacktestEngine

        all_returns = []
        all_win_rates = []
        all_sharpes = []

        for symbol in symbols:
            try:
                data = feed.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    market_hours_only=True,
                )
                if data is None or data.empty:
                    continue

                strategy = strategy_cls()
                engine = SimpleBacktestEngine(initial_capital=25000)
                result = engine.run(strategy, data, symbol=symbol)

                all_returns.append(result.total_return)
                all_win_rates.append(result.win_rate)
                if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio is not None:
                    all_sharpes.append(result.sharpe_ratio)
            except Exception as e:
                logger.warning(f"Backtest failed for {symbol}: {e}")

        if not all_returns:
            return {'sharpe': None, 'win_rate': None, 'total_return': None}

        return {
            'sharpe': sum(all_sharpes) / len(all_sharpes) if all_sharpes else None,
            'win_rate': sum(all_win_rates) / len(all_win_rates),
            'total_return': sum(all_returns) / len(all_returns),
            'symbols_tested': len(all_returns),
        }

    def _build_prompt(self, strategy_name, code, backtest_results, historical_proposals):
        """Build the analysis prompt for Claude."""
        history_text = ""
        if historical_proposals:
            history_text = "\n\n## Previous Proposals (do NOT repeat these)\n"
            for i, p in enumerate(historical_proposals, 1):
                desc = p.get('change_description', p[3] if not isinstance(p, dict) else '')
                status = p.get('status', p[2] if not isinstance(p, dict) else '')
                history_text += f"{i}. [{status}] {desc}\n"

        return f"""You are improving a single intraday US-equities strategy using Robert Carver's "Systematic Trading" discipline. Analyze the REAL backtest results below and propose exactly ONE actionable, verifiable improvement.

## Strategy: {strategy_name}

### Current Code
```python
{code}
```

### Real Backtest Results (from backtest_runs over flat-file/Polygon-Massive data)
{json.dumps(backtest_results, indent=2, default=str)}
{history_text}

## How to diagnose (cite the numbers above)
Classify the regime from the metrics, then tailor the fix to THAT regime:
- NO-SIGNAL: n_signals (signals_count) is ~0 -> the entry never triggers. Fix = loosen/redefine the ENTRY or threshold so signals actually fire. Do NOT tune exits.
- NO-CONVERSION: n_signals > 0 but n_trades is ~0 -> signals fire but never convert to trades (filters/confirmation too strict, or sizing rounds to zero). Fix = relax the conversion gate / confirmation / min-size.
- LOSING-EDGE: n_trades > 0 but win_rate or expectancy/mean_R is negative -> there is activity but no edge. Fix = improve the EXIT, add a FILTER that removes the worst symbols/days, or sharpen the edge. Do NOT just loosen entries.

State which regime applies and quote the exact n_signals, n_trades, win_rate, and expectancy/mean_R you used.

## Hard constraints (reject any proposal that violates these)
1. COST SPEED-LIMIT: round-trip cost is ~20bps (0.1% commission + 0.1% slippage as modeled by SimpleBacktestEngine). Any change that increases turnover MUST show the post-change per-trade edge still clears 20bps; prefer changes that REDUCE turnover or raise edge per trade.
2. SIZING: use volatility-targeted / ATR-based position sizing. NEVER fixed-share sizing and NEVER increase size after a loss (anti-martingale only). If the current code sizes by fixed shares, that is itself a candidate fix.
3. NO OVERFIT: the backtest window is short (see date_span). Do NOT curve-fit to 1-3 specific days. The change must be justifiable out-of-sample and validated across MULTIPLE days/symbols, not a magic constant tuned to the in-sample window.
4. ONE CHANGE ONLY: exactly one old->new parameter change OR one single logic edit — not both, not several.

## expected_improvement must be a numeric pass bar
Name the EXACT metric (expectancy in R / mean_R), the magnitude AND direction (e.g. "expectancy from -0.73R to >= +0.10R"), a MIN-TRADES FLOOR the change must still produce (e.g. ">= 30 trades over the window"), and WHICH symbols/days it should help (cite weak names from per_symbol). This must be concrete enough for an automated backtest to verify. All targets are in R-units / expectancy, never dollar P&L.

## Output (strict JSON only)
```json
{{
    "proposal_type": "param_change" or "code_change",
    "analysis_summary": "Regime classification + the specific numbers (n_signals, n_trades, win_rate, expectancy) that drove it",
    "change_description": "The exact old->new parameter or single logic edit, and why it fits the regime and respects cost/sizing/overfit constraints",
    "expected_improvement": "Exact metric (expectancy/R), magnitude+direction, min-trades floor, and which symbols/days it should help",
    "proposed_code": "...complete Python code for the strategy (full module, not a diff)..."
}}
```"""
