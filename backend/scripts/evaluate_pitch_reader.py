"""
E2 - Pitch reader accuracy: does SimSG read the town, price and category of a pitch correctly?

Runs every labelled pitch in data/evaluation/e2/pitches.csv through:
  - the AI pitch reader (the model in backend/.env), and
  - the rule-based backup (keywords and patterns, no AI),
and checks each field against the hand label.

Labels:
  - price_lo/price_hi: the correct price, or a range of acceptable prices (e.g. "between $4 and $6")
  - no price in the pitch: the app's default price (S$4.50) is the expected answer
  - unsupported town (not one of the 6): the app's default town (Ang Mo Kio) is the expected answer

Run from backend/:
    python scripts/evaluate_pitch_reader.py
Writes data/evaluation/e2/results.csv
"""

import csv
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # so "app" can be imported when running from scripts/

from app.services.query_decomposer import QueryDecomposer, decompose_query_rule_based  # noqa: E402

FOLDER = BACKEND / "data" / "evaluation" / "e2"
FIELDS = ["town", "price", "category"]


def is_correct(label, result):
    """Which of the three fields the reader got right."""
    price = result.candidate_price_sgd
    return {
        "town": result.target_planning_area == label["town"],
        "price": float(label["price_lo"]) - 0.005 <= price <= float(label["price_hi"]) + 0.005,
        "category": result.category_type == label["category"],
    }


def main():
    with (FOLDER / "pitches.csv").open(newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    ai_reader = QueryDecomposer()
    rows = []
    for label in labels:
        for reader in ("ai", "rule_based"):
            if reader == "ai":
                result = ai_reader.decompose(label["pitch"])
                used_ai = bool(getattr(result, "_used_ai", False))
            else:
                result = decompose_query_rule_based(label["pitch"])
                used_ai = False
            correct = is_correct(label, result)
            rows.append({
                "id": label["id"], "type": label["type"], "case": label["case"], "reader": reader,
                "used_ai": used_ai,
                "town": result.target_planning_area, "price": result.candidate_price_sgd,
                "category": result.category_type,
                **{f"{field}_ok": correct[field] for field in FIELDS},
                "all_ok": all(correct.values()),
            })
            print(f"{label['id']} {reader:10} town={result.target_planning_area:12} "
                  f"price={result.candidate_price_sgd:<6} category={result.category_type:20} "
                  f"{'OK' if all(correct.values()) else 'WRONG: ' + ', '.join(k for k, v in correct.items() if not v)}")

    with (FOLDER / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ai_calls = [r for r in rows if r["reader"] == "ai"]
    if not all(r["used_ai"] for r in ai_calls):
        print(f"\nWARNING: the AI reader fell back to rules for "
              f"{sum(not r['used_ai'] for r in ai_calls)} of {len(ai_calls)} pitches")

    print("\n## Accuracy (% of pitches correct)")
    print("reader | pitches | town | price | category | all three")
    for reader in ("ai", "rule_based"):
        for kind in ("normal", "edge", "all"):
            rs = [r for r in rows if r["reader"] == reader and (kind == "all" or r["type"] == kind)]
            pct = lambda key: f"{100 * sum(r[key] for r in rs) / len(rs):.0f}%"
            print(f"{reader} ({kind}) | {len(rs)} | {pct('town_ok')} | {pct('price_ok')} | "
                  f"{pct('category_ok')} | {pct('all_ok')}")
    print(f"\nResults written to {FOLDER / 'results.csv'}")


if __name__ == "__main__":
    main()
