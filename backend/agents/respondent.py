"""
SimSG - respondent agent: each persona answers one market-research question through the OpenAI API.
Integrated with Layer 2 ChromaDB qualitative RAG for local neighborhood context (pricing, competitors, lifestyle).

Try it:
    python -m agents.respondent "Ang Mo Kio" 5 --dry-run                     # print prompts with ChromaDB context, zero API cost
    python -m agents.respondent "Ang Mo Kio" 5 --dry-run --no-rag           # print prompts without ChromaDB context
    python -m agents.respondent "Ang Mo Kio" 5 --seed 42                    # 5 real answers
    python -m agents.respondent "Ang Mo Kio" 5 --seed 42 --provider soclaas # same 5 personas, SoC endpoint
    python -m agents.respondent "Tampines" 30 --question "A new bubble tea shop near your home sells a large milk tea for S$6.50. Would you buy one?"

Results are saved to data/runs/<time>_<area>_<seed>.csv: one row per persona, with its
profile, the exact prompt it saw, its answer, and the tokens used.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]      # .../backend
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    from agents.sampler import sample_personas
except ImportError:
    try:
        from sampler import sample_personas
    except ImportError:
        from backend.agents.sampler import sample_personas

try:                                   # optional: read OPENAI_API_KEY from backend/.env
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
except ImportError:
    pass

# Integration with ChromaDB Layer 2 qualitative RAG singleton
try:
    from app.services.simulation_context import get_simulation_context_service
except ImportError:
    try:
        from backend.app.services.simulation_context import get_simulation_context_service
    except ImportError:
        get_simulation_context_service = None

RUNS = BACKEND / "data" / "runs"

# Which API to call. Set SIMSG_PROVIDER in .env, or pass --provider on the command line.
#   openai  : OPENAI_API_KEY, model from SIMSG_MODEL (default gpt-4.1-mini; avoid gpt-4.1-nano, retired)
#   soclaas : SOCLAAS_BASE_URL, SOCLAAS_API_KEY, SOCLAAS_MODEL (NUS SoC endpoint, OpenAI-compatible)
PROVIDERS = {
    "openai": {"key_var": "OPENAI_API_KEY", "base_url_var": None,
               "model": lambda: os.getenv("SIMSG_MODEL", "gpt-4.1-mini")},
    "soclaas": {"key_var": "SOCLAAS_API_KEY", "base_url_var": "SOCLAAS_BASE_URL",
                "model": lambda: os.getenv("SOCLAAS_MODEL", "default")},
}
PROVIDER = os.getenv("SIMSG_PROVIDER", "openai")
MODEL = PROVIDERS[PROVIDER]["model"]()
CONCURRENCY = int(os.getenv("SIMSG_CONCURRENCY", "8"))  # calls in flight at once

# How the model is asked for JSON. "schema" (strict structured output) is the most reliable;
# if the server rejects it, the code falls back to "object" (JSON mode) and then "prompt" (instructions only).
JSON_MODES = ["schema", "object", "prompt"]
_state = {"mode": os.getenv("SIMSG_JSON_MODE", "schema")}
INCLUDE_ETHNICITY = False                            # off by default: stereotyping risk, no data behind it

DEFAULT_QUESTION = ("A new cafe near your home sells a kopi and kaya toast set for S$7. "
                    "Would you buy it?")

DRIVERS = ["price_or_value", "household_income", "typical_spending", "dwelling",
           "age_or_life_stage", "product_fit", "taste_or_habit", "other"]

SYSTEM_PROMPT = """You are a Singapore resident answering a market-research question.
The facts about you and your local neighborhood context are given below. Answer as a person in that household situation and area would.
Do not invent other facts about yourself (no job, children, hobbies or health unless stated).
The household income is for the whole household, combined, not just you.
The spending figure is the average for households in your type of home, not your own limit,
so you may spend more or less than it.
Consider the local prices, nearby alternatives, and your neighborhood routine alongside what your household can afford.
If the price is too high for your household, or the product does not suit you, say you would not buy.
Answer honestly; most real people say no to most new products.

Fields:
- would_buy: true or false
- max_price_sgd: the most you would pay in Singapore dollars for what is described, or null if you would not buy at any price
- likelihood_1_to_5: 1 = definitely not, 5 = definitely yes
- reason: one sentence, under 25 words, in your own voice
- main_driver: the ONE category that matches the reason you wrote:
    price_or_value      the price compared with what you get
    household_income    what your household can afford
    typical_spending    how it compares with what households like yours usually spend
    dwelling            your type of home (space, rules, facilities)
    age_or_life_stage   your age or stage of life
    product_fit         whether the product itself suits your needs
    taste_or_habit      personal liking, routine or habit
    other               none of the above
- objection: your main concern in a few words, or null if none

Reply with one JSON object containing exactly these six keys and nothing else."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "would_buy": {"type": "boolean"},
        "max_price_sgd": {"type": ["number", "null"]},
        "likelihood_1_to_5": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
        "main_driver": {"type": "string", "enum": DRIVERS},
        "objection": {"type": ["string", "null"]},
    },
    "required": ["would_buy", "max_price_sgd", "likelihood_1_to_5", "reason", "main_driver", "objection"],
    "additionalProperties": False,
}

CATEGORY_LABELS = {
    "pricing_benchmarks": "Local Pricing Norms & Benchmarks",
    "competitors_and_supply": "Nearby Competitors & Food Options",
    "foot_traffic_and_lifestyle": "Neighborhood Lifestyle & Habits",
}


# ---------------------------------------------------------------------------
# ChromaDB Layer 2 RAG integration
# ---------------------------------------------------------------------------

def fetch_rag_context(area: str, query: str, n_results: int = 3) -> List[Any]:
    """
    Retrieve Layer 2 qualitative context chunks from ChromaDB for the area and pitch.
    Returns empty list if ChromaDB is empty or uninitialised.
    """
    if get_simulation_context_service is None:
        return []
    try:
        svc = get_simulation_context_service()
        chunks, _ = svc.query_rag(planning_area=area, user_pitch=query, n_results=n_results)
        return chunks
    except Exception as exc:
        print(f"[note] ChromaDB RAG lookup skipped ({exc})")
        return []


def format_rag_section(rag_chunks: List[Any], area: str) -> str:
    """Format retrieved RAG chunks as readable local neighborhood context."""
    if not rag_chunks:
        return ""
    lines = [f"Local neighborhood context for {area}:"]
    for c in rag_chunks:
        cat = getattr(c, "category", "") or (c.get("category", "") if isinstance(c, dict) else "")
        text = getattr(c, "text", "") or (c.get("text", "") if isinstance(c, dict) else str(c))
        label = CATEGORY_LABELS.get(cat, cat.replace("_", " ").title() if cat else "Local Context")
        lines.append(f"- [{label}]: {text}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Persona -> prompt
# ---------------------------------------------------------------------------

def age_text(group):
    if group.endswith("+"):
        return f"aged {group[:-1]} or over"
    lo, hi = group.split("-")
    return f"aged {lo} to {hi}"


def build_user_prompt(p, question, rag_chunks=None, include_ethnicity=INCLUDE_ETHNICITY):
    who = "woman" if p["sex"] == "Female" else "man"
    eth = f"{p['ethnic_group']} " if include_ethnicity else ""
    rag_section = format_rag_section(rag_chunks, p["area"]) if rag_chunks else ""
    return (
        f"About you: a {eth}{who} {age_text(p['age_group'])}, living in {p['area']}, Singapore, "
        f"in a {p['dwelling_text']}. "
        f"Your household's combined income from work is {p['household_income_text']} a month. "
        f"Households in your type of home spend about S${p['spend_eating_out']:.0f} per person per month "
        f"on eating out, and about S${p['spend_total']:.0f} per person per month in total.\n\n"
        f"{rag_section}"
        f"Question: {question}"
    )


def preview_panel(area: str, n: int, question: str = DEFAULT_QUESTION, seed: Optional[int] = None, use_rag: bool = True):
    """
    Generate panel personas and prompts with ChromaDB context without calling LLM APIs.
    Returns (panel_df, rag_chunks, list_of_prompts, seed).
    """
    panel, seed = sample_personas(area, n, seed)
    rag_chunks = fetch_rag_context(area, question) if use_rag else []
    prompts = [build_user_prompt(p, question, rag_chunks=rag_chunks) for p in panel.to_dict("records")]
    return panel, rag_chunks, prompts, seed


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def parse_json(text):
    """Read the JSON object from a reply, tolerating ```json fences or text around it."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def normalise(a):
    """Fix common type slips from models without strict structured output ('true', '5', '7.50')."""
    if isinstance(a.get("would_buy"), str):
        a["would_buy"] = a["would_buy"].strip().lower() in {"true", "yes", "1"}
    for k, cast in [("likelihood_1_to_5", int), ("max_price_sgd", float)]:
        v = a.get(k)
        if isinstance(v, str):
            v = v.strip().replace("S$", "").replace("$", "")
            a[k] = None if v.lower() in {"", "null", "none"} else cast(float(v))
    if isinstance(a.get("likelihood_1_to_5"), float):
        a["likelihood_1_to_5"] = int(a["likelihood_1_to_5"])
    for k in ["objection", "max_price_sgd"]:
        if isinstance(a.get(k), str) and a[k].strip().lower() in {"null", "none", ""}:
            a[k] = None
    if a.get("would_buy") is False and a.get("max_price_sgd") == 0:
        a["max_price_sgd"] = None      # "0" from a non-buyer means "would not buy at any price"
    return a


def validate(answer):
    """Return a list of problems; empty means the answer is usable."""
    problems = []
    for k in ANSWER_SCHEMA["required"]:
        if k not in answer:
            problems.append(f"missing {k}")
    if answer.get("likelihood_1_to_5") not in {1, 2, 3, 4, 5}:
        problems.append("likelihood out of range")
    mp = answer.get("max_price_sgd")
    if mp is not None and (not isinstance(mp, (int, float)) or mp < 0):
        problems.append("bad max price")
    if answer.get("main_driver") not in DRIVERS:
        problems.append("bad main_driver")
    return problems


async def create(client, messages):
    """Call the API in the current JSON mode; if the server rejects that mode, fall back and retry."""
    while True:
        mode = _state["mode"]
        kwargs = {"model": MODEL, "messages": messages}
        if mode == "schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "survey_answer", "strict": True, "schema": ANSWER_SCHEMA}}
        elif mode == "object":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            return await client.chat.completions.create(**kwargs), mode
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (400, 422) and mode != "prompt":
                if _state["mode"] == mode:
                    _state["mode"] = JSON_MODES[JSON_MODES.index(mode) + 1]
                    print(f"[note] server rejected JSON mode '{mode}' ({str(exc)[:120]}); "
                          f"switching to '{_state['mode']}'")
                continue
            raise


async def ask_one(client, sem, persona, question, rag_chunks=None):
    user_prompt = build_user_prompt(persona, question, rag_chunks=rag_chunks)
    row = {
        "user_prompt": user_prompt,
        "provider": PROVIDER,
        "model": MODEL,
        "rag_chunks_used": len(rag_chunks) if rag_chunks else 0,
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            resp, mode = await create(client, [{"role": "system", "content": SYSTEM_PROMPT},
                                               {"role": "user", "content": user_prompt}])
            row["json_mode"] = mode
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            row["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            row["completion_tokens"] = getattr(usage, "completion_tokens", None)
            if getattr(msg, "refusal", None):
                row["error"] = f"refused: {msg.refusal}"
            else:
                answer = normalise(parse_json(msg.content))
                problems = validate(answer)
                row.update({k: answer.get(k) for k in ANSWER_SCHEMA["required"]})
                row["error"] = "; ".join(problems) if problems else None
        except Exception as exc:          # network, rate limit after retries, bad JSON, etc.
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


async def _run(panel, question, rag_chunks=None):
    from openai import AsyncOpenAI       # imported here so --dry-run works without the package
    cfg = PROVIDERS[PROVIDER]
    key = os.getenv(cfg["key_var"])
    if not key:
        raise SystemExit(f"{cfg['key_var']} is not set. Put it in .env next to this script.")
    base_url = os.getenv(cfg["base_url_var"]) if cfg["base_url_var"] else None
    if cfg["base_url_var"] and not base_url:
        raise SystemExit(f"{cfg['base_url_var']} is not set. Put it in .env next to this script.")
    client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=3, timeout=90)
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [ask_one(client, sem, p, question, rag_chunks=rag_chunks) for p in panel.to_dict("records")]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def income_group(lo):
    if lo < 4000:
        return "1 under $4k"
    if lo < 10000:
        return "2 $4k-$9,999"
    return "3 $10k and over"


def summarise(results):
    ok = results[results["error"].isna()].copy()
    s = {"personas": len(results), "answered": len(ok), "errors": int(results["error"].notna().sum())}
    if ok.empty:
        return s
    ok["would_buy"] = ok["would_buy"].astype(bool)
    share = ok["would_buy"].mean()
    s["share_would_buy"] = round(float(share), 3)
    mp = ok["max_price_sgd"].dropna()
    s["median_max_price"] = float(mp.median()) if len(mp) else None
    s["mean_likelihood"] = round(float(ok["likelihood_1_to_5"].mean()), 2)
    ok["income_group"] = ok["income_lo"].map(income_group)
    s["would_buy_by_income"] = ok.groupby("income_group")["would_buy"].agg(["mean", "count"]).round(2).to_dict("index")
    s["would_buy_by_dwelling"] = ok.groupby("dwelling")["would_buy"].agg(["mean", "count"]).round(2).to_dict("index")
    s["main_driver_counts"] = ok["main_driver"].value_counts().to_dict()
    s["consensus_warning"] = bool(max(share, 1 - share) >= 0.9 and len(ok) >= 10)
    s["prompt_tokens"] = int(results["prompt_tokens"].fillna(0).sum())
    s["completion_tokens"] = int(results["completion_tokens"].fillna(0).sum())
    return s


def use_provider(name):
    """Switch between 'openai' and 'soclaas' at run time."""
    global PROVIDER, MODEL
    if name not in PROVIDERS:
        raise ValueError(f"provider must be one of {list(PROVIDERS)}")
    PROVIDER, MODEL = name, PROVIDERS[name]["model"]()


def run_panel(area, n, question=DEFAULT_QUESTION, seed=None, provider=None, use_rag=True):
    """Sample n personas for the area, ask each the question with ChromaDB context, save and return (results, summary)."""
    if provider:
        use_provider(provider)
    panel, seed = sample_personas(area, n, seed)
    rag_chunks = fetch_rag_context(area, question) if use_rag else []
    answers = asyncio.run(_run(panel, question, rag_chunks=rag_chunks))
    results = pd.concat([panel.reset_index(drop=True), pd.DataFrame(answers)], axis=1)
    results["question"] = question
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{area.replace(' ', '')}_{seed}.csv"
    results.to_csv(path, index=False)
    summary = summarise(results)
    rag_categories = [getattr(c, "category", "") or (c.get("category", "") if isinstance(c, dict) else "") for c in rag_chunks]
    summary.update({
        "area": area,
        "seed": seed,
        "provider": PROVIDER,
        "model": MODEL,
        "json_mode": _state["mode"],
        "rag_enabled": bool(use_rag and len(rag_chunks) > 0),
        "rag_chunks_count": len(rag_chunks),
        "rag_categories": rag_categories,
        "file": str(path),
    })
    return results, summary


def main():
    ap = argparse.ArgumentParser(description="SimSG - LLM Focus Group Simulation with ChromaDB RAG")
    ap.add_argument("area", nargs="?", default="Ang Mo Kio", help="Singapore Planning Area (e.g. 'Ang Mo Kio', 'Tampines')")
    ap.add_argument("n", nargs="?", type=int, default=5, help="Number of personas to sample (default: 5)")
    ap.add_argument("--question", default=DEFAULT_QUESTION, help="Market-research question or pitch")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible personas")
    ap.add_argument("--dry-run", action="store_true", help="Print personas and prompts with ChromaDB context; no LLM API calls")
    ap.add_argument("--no-rag", action="store_true", help="Disable ChromaDB Layer 2 qualitative context")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None,
                    help="openai or soclaas (default: SIMSG_PROVIDER in .env, else openai)")
    args = ap.parse_args()
    if args.provider:
        use_provider(args.provider)

    use_rag = not args.no_rag

    if args.dry_run:
        panel, rag_chunks, prompts, seed = preview_panel(args.area, args.n, args.question, args.seed, use_rag=use_rag)
        rag_info = f"enabled ({len(rag_chunks)} chunks)" if use_rag and rag_chunks else ("empty/unavailable" if use_rag else "disabled")
        print(f"Seed {seed} | provider {PROVIDER} | model {MODEL} | ChromaDB RAG: {rag_info}\n")
        print(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n")
        if rag_chunks:
            print(f"Retrieved {len(rag_chunks)} ChromaDB RAG chunks for {args.area}:")
            for chunk in rag_chunks:
                cat = getattr(chunk, "category", "") or (chunk.get("category", "") if isinstance(chunk, dict) else "")
                dist = getattr(chunk, "cosine_distance", None)
                dist_str = f" (dist={dist})" if dist is not None else ""
                print(f"  - [{cat}]{dist_str}")
            print()
        for p, prompt in zip(panel.to_dict("records"), prompts):
            print(f"--- {p['persona_id']}\n{prompt}\n")
        return

    results, summary = run_panel(args.area, args.n, args.question, args.seed, use_rag=use_rag)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    cols = ["persona_id", "age_group", "dwelling", "household_income_text",
            "would_buy", "max_price_sgd", "likelihood_1_to_5", "main_driver", "reason", "error"]
    print(results[[c for c in cols if c in results]].to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    if summary.get("consensus_warning"):
        print("\nWARNING: 90% or more gave the same answer. Treat this as a sign the panel may be too uniform.")


if __name__ == "__main__":
    main()
