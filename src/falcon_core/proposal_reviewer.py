"""
Proposal Review Agent

Automatically reviews AI advisor proposals by:
1. Loading pending/approved proposals
2. Backtesting the proposed code against the current code
3. Comparing metrics (Sharpe, win rate, return)
4. Auto-approving, auto-rejecting, or flagging for human review
5. Applying approved proposals to the strategy roster
6. Generating a review report

Thresholds:
  - AUTO-APPROVE: proposed Sharpe > current AND win rate improves or holds
  - AUTO-REJECT: proposed Sharpe worse AND win rate worse
  - FLAG: mixed results (one metric up, another down)

Usage:
    from falcon_core.proposal_reviewer import ProposalReviewer
    reviewer = ProposalReviewer(db)
    report = reviewer.review_all()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of reviewing a single proposal."""
    proposal_id: int
    strategy_name: str
    decision: str  # 'approved', 'rejected', 'flagged'
    reason: str
    current_metrics: Dict[str, float] = field(default_factory=dict)
    proposed_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    applied: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewReport:
    """Summary report of a review cycle."""
    timestamp: str
    total_reviewed: int = 0
    approved: int = 0
    rejected: int = 0
    flagged: int = 0
    applied: int = 0
    errors: int = 0
    results: List[ReviewResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'total_reviewed': self.total_reviewed,
            'approved': self.approved,
            'rejected': self.rejected,
            'flagged': self.flagged,
            'applied': self.applied,
            'errors': self.errors,
            'results': [r.to_dict() for r in self.results],
        }


class ProposalReviewer:
    """
    Reviews advisor proposals by backtesting proposed code and comparing metrics.
    """

    # Thresholds for auto-decision
    MIN_TRADES_FOR_DECISION = 3  # Need at least this many trades to judge
    SHARPE_IMPROVEMENT_THRESHOLD = 0.1  # Sharpe must improve by at least this
    WIN_RATE_DECLINE_TOLERANCE = 0.05  # Allow up to 5% win rate decline if Sharpe improves

    def __init__(self, db_manager, auto_apply: bool = True):
        """
        Args:
            db_manager: DatabaseManager instance
            auto_apply: If True, automatically apply approved proposals to roster
        """
        self.db = db_manager
        self.auto_apply = auto_apply

    def review_all(self, statuses: List[str] = None) -> ReviewReport:
        """
        Review all proposals with the given statuses.

        Args:
            statuses: List of proposal statuses to review.
                      Default: ['pending', 'approved'] — approved proposals
                      that haven't been backtested yet.

        Returns:
            ReviewReport with all results.
        """
        if statuses is None:
            statuses = ['pending', 'approved']

        now = datetime.now().isoformat()
        report = ReviewReport(timestamp=now)

        # Get proposals to review
        placeholders = ','.join(['%s'] * len(statuses))
        proposals = self.db.execute(
            f"""SELECT id, strategy_name, proposal_type, current_code, proposed_code,
                       analysis_summary, change_description, expected_improvement,
                       current_sharpe, proposed_sharpe, status
                FROM strategy_proposals
                WHERE status IN ({placeholders})
                  AND proposed_code IS NOT NULL
                ORDER BY created_at DESC""",
            tuple(statuses), fetch='all',
        )

        if not proposals:
            logger.info("No proposals to review")
            return report

        logger.info(f"Reviewing {len(proposals)} proposals")

        for row in proposals:
            proposal_id = row['id']
            strategy_name = row['strategy_name']
            report.total_reviewed += 1

            try:
                result = self._review_one(row)
                report.results.append(result)

                if result.decision == 'approved':
                    report.approved += 1
                    if self.auto_apply and result.applied:
                        report.applied += 1
                elif result.decision == 'rejected':
                    report.rejected += 1
                elif result.decision == 'flagged':
                    report.flagged += 1

                if result.error:
                    report.errors += 1

            except Exception as e:
                logger.error(f"Error reviewing proposal {proposal_id} ({strategy_name}): {e}")
                report.errors += 1
                report.results.append(ReviewResult(
                    proposal_id=proposal_id,
                    strategy_name=strategy_name,
                    decision='flagged',
                    reason=f"Review error: {e}",
                    error=str(e),
                ))

        # Store the report
        self._store_report(report)

        logger.info(
            f"Review complete: {report.approved} approved, {report.rejected} rejected, "
            f"{report.flagged} flagged, {report.applied} applied, {report.errors} errors"
        )

        return report

    def _review_one(self, row) -> ReviewResult:
        """Review a single proposal by backtesting both versions."""
        proposal_id = row['id']
        strategy_name = row['strategy_name']
        current_code = row['current_code']
        proposed_code = row['proposed_code']

        logger.info(f"Reviewing proposal #{proposal_id}: {strategy_name}")

        # Get strategy config from roster
        roster_row = self.db.execute(
            'SELECT symbols, interval FROM strategy_roster WHERE strategy_name = %s',
            (strategy_name,), fetch='one',
        )

        if not roster_row:
            return ReviewResult(
                proposal_id=proposal_id,
                strategy_name=strategy_name,
                decision='flagged',
                reason=f"Strategy '{strategy_name}' not found in roster",
            )

        symbols = roster_row['symbols']
        if isinstance(symbols, str):
            symbols = json.loads(symbols)
        interval = roster_row['interval']

        # Backtest current code
        current_metrics = self._backtest_code(current_code, strategy_name, symbols, interval)

        # Backtest proposed code
        proposed_metrics = self._backtest_code(proposed_code, strategy_name, symbols, interval)

        if current_metrics.get('error') or proposed_metrics.get('error'):
            return ReviewResult(
                proposal_id=proposal_id,
                strategy_name=strategy_name,
                decision='flagged',
                reason=f"Backtest error — current: {current_metrics.get('error')}, proposed: {proposed_metrics.get('error')}",
                current_metrics=current_metrics,
                proposed_metrics=proposed_metrics,
                error=current_metrics.get('error') or proposed_metrics.get('error'),
            )

        # Compare metrics
        improvement = {
            'sharpe': proposed_metrics.get('sharpe', 0) - current_metrics.get('sharpe', 0),
            'win_rate': proposed_metrics.get('win_rate', 0) - current_metrics.get('win_rate', 0),
            'total_return': proposed_metrics.get('total_return', 0) - current_metrics.get('total_return', 0),
            'total_trades': proposed_metrics.get('total_trades', 0) - current_metrics.get('total_trades', 0),
        }

        decision, reason = self._make_decision(current_metrics, proposed_metrics, improvement)

        result = ReviewResult(
            proposal_id=proposal_id,
            strategy_name=strategy_name,
            decision=decision,
            reason=reason,
            current_metrics=current_metrics,
            proposed_metrics=proposed_metrics,
            improvement=improvement,
        )

        # Update proposal in DB
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE strategy_proposals SET
               status = %s,
               reviewed_by = %s,
               reviewed_at = %s,
               review_notes = %s,
               current_sharpe = %s,
               proposed_sharpe = %s,
               current_win_rate = %s,
               proposed_win_rate = %s,
               current_total_return = %s,
               proposed_total_return = %s
               WHERE id = %s""",
            (
                decision, 'proposal-reviewer', now, reason,
                current_metrics.get('sharpe', 0),
                proposed_metrics.get('sharpe', 0),
                current_metrics.get('win_rate', 0),
                proposed_metrics.get('win_rate', 0),
                current_metrics.get('total_return', 0),
                proposed_metrics.get('total_return', 0),
                proposal_id,
            ),
        )

        # Auto-apply if approved
        if decision == 'approved' and self.auto_apply:
            applied = self._apply_proposal(proposal_id, strategy_name, proposed_code, proposed_metrics)
            result.applied = applied

        return result

    def _backtest_code(self, code: str, name: str, symbols: list,
                       interval: str) -> Dict[str, Any]:
        """Backtest strategy code and return aggregate metrics."""
        from falcon_core.backtesting.strategy_loader import load_strategy_from_code, validate_strategy_code
        from falcon_core.backtesting.data_feed import DataFeed
        from falcon_core.backtesting.engine import SimpleBacktestEngine
        from datetime import date, timedelta

        # Validate code first
        is_valid, reason = validate_strategy_code(code)
        if not is_valid:
            return {'error': f"Code validation failed: {reason}"}

        try:
            StrategyClass = load_strategy_from_code(code, name)
            if StrategyClass is None:
                return {'error': 'No strategy class found in code'}
        except Exception as e:
            return {'error': f"Failed to load: {e}"}

        feed = DataFeed(db_manager=self.db)
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=42)

        all_returns = []
        all_win_rates = []
        all_sharpes = []
        total_trades = 0

        for symbol in symbols[:3]:  # Cap at 3 symbols for speed
            try:
                data = feed.get_historical_data(
                    symbol, start_date, end_date,
                    interval=interval, market_hours_only=True,
                )
                if data is None or data.empty:
                    continue

                strategy = StrategyClass()
                engine = SimpleBacktestEngine(initial_capital=25000)
                result = engine.run(strategy, data, symbol=symbol)

                all_returns.append(result.total_return)
                all_win_rates.append(result.win_rate)
                if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio is not None:
                    all_sharpes.append(result.sharpe_ratio)
                total_trades += result.total_trades

            except Exception as e:
                logger.warning(f"  Backtest failed for {symbol}: {e}")

        if not all_returns:
            return {
                'sharpe': 0.0, 'win_rate': 0.0, 'total_return': 0.0,
                'total_trades': 0, 'symbols_tested': 0,
            }

        return {
            'sharpe': sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0,
            'win_rate': sum(all_win_rates) / len(all_win_rates),
            'total_return': sum(all_returns) / len(all_returns),
            'total_trades': total_trades,
            'symbols_tested': len(all_returns),
        }

    def _make_decision(self, current: dict, proposed: dict,
                       improvement: dict) -> tuple:
        """
        Decide: approved, rejected, or flagged.

        Returns (decision, reason)
        """
        cur_trades = current.get('total_trades', 0)
        prop_trades = proposed.get('total_trades', 0)

        # If current has zero trades and proposed has some, approve
        if cur_trades == 0 and prop_trades > 0:
            return 'approved', (
                f"Proposed generates {prop_trades} trades where current generates none "
                f"(return={proposed['total_return']:.2%}, wr={proposed['win_rate']:.1%})"
            )

        # If neither generates trades, flag for human review
        if cur_trades == 0 and prop_trades == 0:
            return 'flagged', "Neither current nor proposed code generates trades"

        # If proposed kills all trades, reject
        if cur_trades > 0 and prop_trades == 0:
            return 'rejected', f"Proposed generates zero trades (current had {cur_trades})"

        # Both have trades — compare metrics
        sharpe_delta = improvement.get('sharpe', 0)
        wr_delta = improvement.get('win_rate', 0)
        return_delta = improvement.get('total_return', 0)

        # Clear improvement: Sharpe up and win rate not significantly worse
        if sharpe_delta > self.SHARPE_IMPROVEMENT_THRESHOLD and wr_delta >= -self.WIN_RATE_DECLINE_TOLERANCE:
            return 'approved', (
                f"Sharpe improved by {sharpe_delta:+.2f} "
                f"(wr {wr_delta:+.1%}, return {return_delta:+.2%})"
            )

        # Clear degradation: both Sharpe and win rate worse
        if sharpe_delta < -self.SHARPE_IMPROVEMENT_THRESHOLD and wr_delta < -self.WIN_RATE_DECLINE_TOLERANCE:
            return 'rejected', (
                f"Both metrics worse: Sharpe {sharpe_delta:+.2f}, wr {wr_delta:+.1%}"
            )

        # Return improvement even if Sharpe is flat
        if return_delta > 0.01 and wr_delta >= 0:
            return 'approved', (
                f"Return improved by {return_delta:+.2%} "
                f"(sharpe {sharpe_delta:+.2f}, wr {wr_delta:+.1%})"
            )

        # Mixed results — flag for human review
        return 'flagged', (
            f"Mixed results: sharpe {sharpe_delta:+.2f}, wr {wr_delta:+.1%}, "
            f"return {return_delta:+.2%} — needs human review"
        )

    def _apply_proposal(self, proposal_id: int, strategy_name: str,
                        proposed_code: str, metrics: dict) -> bool:
        """Apply an approved proposal: update strategy_roster with new code and metrics."""
        try:
            now = datetime.now().isoformat()

            self.db.execute(
                """UPDATE strategy_roster SET
                   strategy_code = %s,
                   backtest_sharpe = %s,
                   backtest_win_rate = %s,
                   backtest_total_return = %s,
                   last_backtest_at = %s,
                   updated_at = %s
                   WHERE strategy_name = %s""",
                (
                    proposed_code,
                    metrics.get('sharpe', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('total_return', 0),
                    now, now, strategy_name,
                ),
            )

            self.db.execute(
                "UPDATE strategy_proposals SET applied_at = %s WHERE id = %s",
                (now, proposal_id),
            )

            # Log the rotation
            self.db.execute(
                """INSERT INTO strategy_rotation_log
                   (strategy_name, from_status, to_status, reason, rotated_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (strategy_name, 'backtest', 'backtest',
                 f'Applied proposal #{proposal_id}', now),
            )

            logger.info(f"  Applied proposal #{proposal_id} to {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"  Failed to apply proposal #{proposal_id}: {e}")
            return False

    def _store_report(self, report: ReviewReport):
        """Store review report in agent_memory for RAG access."""
        try:
            summary = (
                f"Proposal review: {report.approved} approved, "
                f"{report.rejected} rejected, {report.flagged} flagged, "
                f"{report.applied} applied"
            )

            details = []
            for r in report.results:
                details.append(
                    f"{r.strategy_name}: {r.decision} — {r.reason}"
                )

            content = summary + "\n\n" + "\n".join(details)

            self.db.execute(
                """INSERT INTO agent_memory
                   (agent_name, category, content, metadata, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    'proposal-reviewer',
                    'review-report',
                    content,
                    json.dumps(report.to_dict()),
                    report.timestamp,
                    report.timestamp,
                ),
            )
        except Exception as e:
            logger.warning(f"Could not store review report in agent_memory: {e}")
