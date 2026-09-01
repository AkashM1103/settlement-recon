"""CLI entrypoint: run the reconciliation, print the report, optionally ask questions.

    python main.py                        # full report on the bundled sample batch
    python main.py --ask "total tax?"     # report + one grounded question
    python main.py --chat                 # interactive Q&A loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recon import MatchConfig, SAMPLE_QUESTIONS, reconcile_paths, render_text_report  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orders", type=Path, default=DATA_DIR / "orders.csv")
    p.add_argument("--settlements", type=Path, default=DATA_DIR / "settlements.csv")
    p.add_argument("--ground-truth", type=Path, default=DATA_DIR / "ground_truth.csv")
    p.add_argument("--amount-tolerance", type=float, default=MatchConfig.amount_tolerance)
    p.add_argument("--date-window", type=int, default=MatchConfig.date_window_days)
    p.add_argument("--accept-confidence", type=float, default=MatchConfig.accept_confidence)
    p.add_argument("--no-llm", action="store_true", help="force the deterministic reasoner")
    p.add_argument("--ask", action="append", default=[], help="question to answer after the report")
    p.add_argument("--demo-questions", action="store_true", help="answer the built-in sample questions")
    p.add_argument("--chat", action="store_true", help="interactive Q&A loop")
    p.add_argument("--out", type=Path, default=OUT_DIR, help="directory for CSV artifacts")
    args = p.parse_args()

    config = MatchConfig(
        amount_tolerance=args.amount_tolerance,
        date_window_days=args.date_window,
        accept_confidence=args.accept_confidence,
    )
    result = reconcile_paths(
        args.orders, args.settlements, args.ground_truth, config=config, use_llm=not args.no_llm
    )

    print(render_text_report(result.metrics, result.breakdown, result.money))

    args.out.mkdir(parents=True, exist_ok=True)
    result.matches.to_csv(args.out / "reconciliation.csv", index=False)
    result.breakdown.to_csv(args.out / "exceptions.csv", index=False)
    result.outcome.unsettled_orders.to_csv(args.out / "unsettled_orders.csv", index=False)
    print(f"\nartifacts written to {args.out}/")

    questions = list(args.ask) + (SAMPLE_QUESTIONS if args.demo_questions else [])
    if questions or args.chat:
        qa = result.qa()
        for question in questions:
            answer = qa.ask(question)
            print(f"\nQ: {question}\nA: {answer.answer}\n   cited: {', '.join(answer.cited_records[:12]) or 'none'}")
        if args.chat:
            print("\nQ&A mode - blank line or 'exit' to quit.")
            while True:
                try:
                    question = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question or question.lower() in {"exit", "quit"}:
                    break
                answer = qa.ask(question)
                print(f"{answer.answer}\n   cited: {', '.join(answer.cited_records[:12]) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
