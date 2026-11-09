"""
E4 - Persona response behaviour: are the AI personas' answers coherent, sensible and stable?

Runs one pitch (the S$7 artisan kopi bar in Ang Mo Kio) through the full app pipeline (/api/simulate):
  - price sweep: S$3, 5, 7, 9, 12 with the same 30 personas (seed 2026)
  - repeats:     3 more runs at S$7 with the same personas (only the AI's randomness changes)
  - new panels:  2 runs at S$7 with different personas (seeds 2027, 2028)
Each response is saved to data/evaluation/e4/; runs already saved are not repeated (delete the
JSON files to rerun). Then it prints:
  1. demand curve:   mean purchase likelihood at each price (should fall as price rises)
  2. coherence:      price limits in order (too cheap <= cheap <= expensive <= too expensive), and low
                     purchase likelihood when the price is above a persona's "too expensive"
  3. income check:   median "expensive" limit by household income (should rise with income)
  4. anchoring:      personas give price limits before seeing the price, so the limits should not
                     follow the seller's price across the sweep
  5. stability:      mean purchase likelihood and optimal price point across repeat runs

Needs an API key in backend/.env (the AI personas). About 60 AI calls per run.
Run from backend/:
    python scripts/evaluate_responses.py
"""

import json
import statistics as st
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # so "app" can be imported when running from scripts/

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.rate_limiter import rate_limiter  # noqa: E402

OUT = BACKEND / "data" / "evaluation" / "e4"
PITCH = "Artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"
SEED = 2026
SWEEP_PRICES = [3, 5, 7, 9, 12]
RUNS = ([(f"price_{p}", p, SEED) for p in SWEEP_PRICES]
        + [(f"repeat_{i}", 7, SEED) for i in (1, 2, 3)]
        + [(f"seed_{s}", 7, s) for s in (2027, 2028)])


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    for name, price, seed in RUNS:
        path = OUT / f"{name}.json"
        if path.exists():
            continue
        rate_limiter.reset_all()  # this script is the only user; don't let the per-session limit stop it
        body = {"raw_query": PITCH, "override_price": price, "override_town": "Ang Mo Kio",
                "sample_size": 30, "seed": seed, "use_mock": False}
        response = client.post("/api/simulate", json=body, headers={"X-Session-ID": "evaluation"})
        response.raise_for_status()
        path.write_text(json.dumps(response.json(), indent=1), encoding="utf-8")
        print(f"saved {name} (S${price}, seed {seed}): ai_usage={response.json().get('ai_usage')}")


def load(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def income_group(bracket, amount):
    """below S$3k / S$3k-9,999 / S$10k+ from the census band text (or the amount if no band)."""
    lo = amount
    if bracket:
        text = bracket.replace("S$", "").replace(",", "")
        lo = 0 if text.startswith("below") else float(text.split()[0])
    return "below S$3,000" if lo < 3000 else ("S$3,000-9,999" if lo < 10000 else "S$10,000+")


def main():
    run_all()
    runs = {name: load(name) for name, _, _ in RUNS}

    # Check every persona really used the AI
    backups = sum((r.get("ai_usage") or {}).get("persona_backup", 0) for r in runs.values())
    print(f"\nPersonas that used the rule-based backup instead of the AI: {backups}")

    print("\n## 1. Demand curve (same 30 personas)")
    print("price | mean purchase likelihood | personas >= 50% | optimal price point (OPP)")
    for p in SWEEP_PRICES:
        r = runs[f"price_{p}"]
        buys = [x["buy_probability_pct"] for x in r["personas"]]
        print(f"S${p} | {st.mean(buys):.1f}% | {sum(b >= 50 for b in buys)}/{len(buys)} | "
              f"S${r['statistics']['psm_intersections']['opp']:.2f}")

    print("\n## 2. Coherence (all runs)")
    personas = [(x, price) for name, price, _ in RUNS for x in runs[name]["personas"]]
    limits = [x for x, _ in personas if None not in (x["too_cheap"], x["cheap"], x["expensive"], x["too_expensive"])]
    ordered = [x for x in limits if x["too_cheap"] <= x["cheap"] <= x["expensive"] <= x["too_expensive"]]
    print(f"price limits in order: {len(ordered)}/{len(limits)} ({100 * len(ordered) / len(limits):.1f}%)")
    above = [(x, price) for x, price in personas if x["too_expensive"] is not None and price > x["too_expensive"]]
    low = [x for x, _ in above if x["buy_probability_pct"] <= 20]
    if above:
        print(f"price above the persona's 'too expensive': {len(above)} answers, "
              f"{len(low)} ({100 * len(low) / len(above):.0f}%) with purchase likelihood <= 20%")

    print("\n## 3. Income check (S$7 runs with seed 2026, pooled)")
    same = [x for name in ("price_7", "repeat_1", "repeat_2", "repeat_3") for x in runs[name]["personas"]]
    print("income group | answers | median 'expensive' | median 'too expensive' | mean purchase likelihood")
    for g in ("below S$3,000", "S$3,000-9,999", "S$10,000+"):
        xs = [x for x in same if income_group(x.get("income_bracket"), x["monthly_income_sgd"]) == g]
        if xs:
            print(f"{g} | {len(xs)} | S${st.median(x['expensive'] for x in xs):.2f} | "
                  f"S${st.median(x['too_expensive'] for x in xs):.2f} | "
                  f"{st.mean(x['buy_probability_pct'] for x in xs):.1f}%")

    print("\n## 4. Anchoring: price limits across the sweep (same personas)")
    print("seller's price | median 'cheap' | median 'expensive' | median 'too expensive'")
    for p in SWEEP_PRICES:
        xs = runs[f"price_{p}"]["personas"]
        print(f"S${p} | S${st.median(x['cheap'] for x in xs):.2f} | S${st.median(x['expensive'] for x in xs):.2f} | "
              f"S${st.median(x['too_expensive'] for x in xs):.2f}")
    same_limit = sum(1 for p in SWEEP_PRICES for x in runs[f"price_{p}"]["personas"]
                     if p in (x["expensive"], x["too_expensive"]))
    print(f"answers whose 'expensive' or 'too expensive' equals the seller's price: {same_limit}/"
          f"{30 * len(SWEEP_PRICES)}")

    print("\n## 5. Stability at S$7")
    print("run | personas | mean purchase likelihood | OPP")
    groups = {"same personas (seed 2026)": ["price_7", "repeat_1", "repeat_2", "repeat_3"],
              "different personas": ["price_7", "seed_2027", "seed_2028"]}
    for label, names in groups.items():
        buys = [runs[n]["statistics"]["mean_buy_intent_pct"] for n in names]
        opps = [runs[n]["statistics"]["psm_intersections"]["opp"] for n in names]
        for n, b, o in zip(names, buys, opps):
            print(f"{n} | {label} | {b:.1f}% | S${o:.2f}")
        print(f"  -> {label}: purchase likelihood mean {st.mean(buys):.1f}%, range {min(buys):.1f}-{max(buys):.1f}, "
              f"SD {st.stdev(buys):.1f}; OPP range S${min(opps):.2f}-{max(opps):.2f}")


if __name__ == "__main__":
    main()
