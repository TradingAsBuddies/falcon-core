"""
CLI for falcon-proposal-review.

Usage:
    falcon-proposal-review                 # Review all pending/approved proposals
    falcon-proposal-review --dry-run       # Review without applying
    falcon-proposal-review --json          # Machine-readable output
    falcon-proposal-review --report        # Show last review report
"""

import argparse
import json
import logging
import sys
import os


def _load_env():
    """Load falcon.env."""
    for path in ['/etc/falcon/falcon.env', os.path.expanduser('~/.config/falcon/falcon.env')]:
        if os.path.exists(path):
            try:
                from dotenv import load_dotenv
                load_dotenv(path, override=False)
            except ImportError:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, _, v = line.partition('=')
                            if not os.getenv(k.strip()):
                                os.environ[k.strip()] = v.strip()
            return


def main():
    parser = argparse.ArgumentParser(
        prog='falcon-proposal-review',
        description='Review and apply AI advisor strategy proposals',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Review without applying changes')
    parser.add_argument('--json', '-j', action='store_true', dest='json_output',
                        help='Output results as JSON')
    parser.add_argument('--report', '-r', action='store_true',
                        help='Show the last review report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    _load_env()

    from falcon_core.db_manager import get_db_manager
    db = get_db_manager()

    if args.report:
        _show_last_report(db, args.json_output)
        return

    from falcon_core.proposal_reviewer import ProposalReviewer

    reviewer = ProposalReviewer(db, auto_apply=not args.dry_run)
    report = reviewer.review_all()

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        _print_report(report, dry_run=args.dry_run)

    # Exit code: 1 if any errors
    if report.errors > 0:
        sys.exit(1)


def _print_report(report, dry_run=False):
    mode = " (DRY RUN)" if dry_run else ""
    print(f"\n{'='*70}")
    print(f"  PROPOSAL REVIEW REPORT{mode}")
    print(f"{'='*70}")
    print(f"  Reviewed: {report.total_reviewed}")
    print(f"  Approved: {report.approved}  |  Rejected: {report.rejected}  |  Flagged: {report.flagged}")
    if not dry_run:
        print(f"  Applied:  {report.applied}")
    if report.errors:
        print(f"  Errors:   {report.errors}")
    print(f"{'='*70}\n")

    for r in report.results:
        status_sym = {'approved': '+', 'rejected': '-', 'flagged': '?'}.get(r.decision, '?')
        applied_str = " [APPLIED]" if r.applied else ""
        print(f"  [{status_sym}] {r.strategy_name}{applied_str}")
        print(f"      {r.reason}")

        if r.current_metrics and r.proposed_metrics:
            cm = r.current_metrics
            pm = r.proposed_metrics
            print(f"      Current:  sharpe={cm.get('sharpe',0):>+6.2f}  wr={cm.get('win_rate',0):>5.1%}  "
                  f"return={cm.get('total_return',0):>+6.2%}  trades={cm.get('total_trades',0)}")
            print(f"      Proposed: sharpe={pm.get('sharpe',0):>+6.2f}  wr={pm.get('win_rate',0):>5.1%}  "
                  f"return={pm.get('total_return',0):>+6.2%}  trades={pm.get('total_trades',0)}")
        print()

    if not report.results:
        print("  No proposals to review.\n")


def _show_last_report(db, json_output):
    """Show the most recent review report from agent_memory."""
    row = db.execute(
        """SELECT content, metadata, created_at FROM agent_memory
           WHERE agent_name = 'proposal-reviewer' AND category = 'review-report'
           ORDER BY created_at DESC LIMIT 1""",
        fetch='one',
    )

    if not row:
        print("No review reports found.")
        return

    if json_output:
        meta = row['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta)
        print(json.dumps(meta, indent=2, default=str))
    else:
        print(f"\nLast review: {row['created_at']}\n")
        print(row['content'])


if __name__ == '__main__':
    main()
