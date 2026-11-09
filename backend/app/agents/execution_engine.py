"""
backend/app/agents/execution_engine.py
----------------------------------------
Async Batch Persona Execution Engine for SimSG:
Evaluates synthetic cohorts of Singapore residents (30 personas) against a
decomposed business pitch and Layer 2 ChromaDB qualitative neighborhood context.

Core Capabilities:
  - Bounded concurrent execution using asyncio.gather and asyncio.Semaphore(25).
  - Uses the provider/model from backend/.env (SIMSG_PROVIDER, SIMSG_MODEL) with structured JSON output enforcement.
  - Two-part interview per persona, so price limits are not anchored on the seller's price:
      1. price limits: concept with prices hidden -> 4 Van Westendorp thresholds
      2. purchase: same conversation, price revealed -> buy %, sentiment, quote, objection
  - Contextual prompt conditioning across demographic attributes, typical eating-out spend,
    and local neighborhood conditions (including local prices from ChromaDB).
  - Monotonicity post-processing enforcing Van Westendorp price thresholds:
      too_cheap <= cheap < expensive <= too_expensive
  - Resilient offline fallback: Deterministic rule-based evaluation when API key is
    missing, dry-run is requested, or network errors occur.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr, field_validator

logger = logging.getLogger("simsg.execution_engine")

# ---------------------------------------------------------------------------
# Path & Environment Resolution
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent.parent  # backend/
_ENV_PATH = _BACKEND_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


# ---------------------------------------------------------------------------
# Schema Imports (Robust across package relative and absolute paths)
# ---------------------------------------------------------------------------
try:
    from app.services.sampler import SyntheticPersona, average_daily_eating_out_sgd, living_arrangement_text
    from app.services.query_decomposer import DecomposedConcept
except ImportError:
    from backend.app.services.sampler import SyntheticPersona, average_daily_eating_out_sgd, living_arrangement_text
    from backend.app.services.query_decomposer import DecomposedConcept

# Provider / model come from backend/.env (SIMSG_PROVIDER, SIMSG_MODEL, ...)
try:
    from app.services.llm_settings import describe_missing_provider, get_llm_settings
except ImportError:
    from backend.app.services.llm_settings import describe_missing_provider, get_llm_settings


# ---------------------------------------------------------------------------
# Pydantic Output Model
# ---------------------------------------------------------------------------
class PersonaEvaluationOutput(BaseModel):
    """
    Individual synthetic resident evaluation output for a business concept pitch.
    """

    # True only when both AI calls answered (False = rule-based backup). Not part of the data or JSON schema.
    _used_ai: bool = PrivateAttr(default=False)

    persona_id: str = Field(..., description="Unique persona identifier (e.g. 'AMK-001')")
    buy_probability_pct: int = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated purchase likelihood percentage from 0 to 100",
    )
    too_cheap: float = Field(
        ...,
        ge=0.0,
        description="Van Westendorp price threshold: so cheap quality is questioned",
    )
    cheap: float = Field(
        ...,
        ge=0.0,
        description="Van Westendorp price threshold: considered a bargain / good value",
    )
    expensive: float = Field(
        ...,
        ge=0.0,
        description="Van Westendorp price threshold: considered expensive but still considered",
    )
    too_expensive: float = Field(
        ...,
        ge=0.0,
        description="Van Westendorp price threshold: so expensive customer refuses to purchase",
    )
    sentiment_label: Literal["Positive", "Neutral", "Negative"] = Field(
        ...,
        description="Overall persona sentiment: Positive, Neutral, or Negative",
    )
    in_character_quote: str = Field(
        ...,
        description="Authentic reaction quote in standard Singapore English",
    )
    primary_objection: str = Field(
        ...,
        max_length=120,
        description="Concise primary objection or friction point (max 120 chars)",
    )

    @field_validator("primary_objection", mode="before")
    @classmethod
    def clamp_primary_objection(cls, v: Any) -> str:
        s = str(v).strip() if v is not None else "None"
        return s[:120]

    @field_validator("buy_probability_pct", mode="before")
    @classmethod
    def clamp_buy_prob(cls, v: Any) -> int:
        try:
            val = int(round(float(v)))
            return max(0, min(100, val))
        except Exception:
            return 50

    @field_validator("sentiment_label", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> str:
        s = str(v).strip().capitalize() if v else "Neutral"
        if s in ("Positive", "Neutral", "Negative"):
            return s
        lower = str(v).lower()
        if any(w in lower for w in ("pos", "good", "like", "love", "yes", "great", "support")):
            return "Positive"
        if any(w in lower for w in ("neg", "bad", "hate", "no", "expensive", "ex", "poor", "siao")):
            return "Negative"
        return "Neutral"


# ---------------------------------------------------------------------------
# Monotonicity Post-Processing Function
# ---------------------------------------------------------------------------
def repair_price_monotonicity(
    too_cheap: float,
    cheap: float,
    expensive: float,
    too_expensive: float,
    min_step: float = 0.20,
) -> Tuple[float, float, float, float]:
    """
    Enforces the Van Westendorp Price Sensitivity Meter monotonicity constraint:
        too_cheap <= cheap < expensive <= too_expensive

    Repairs inversions, ties between cheap and expensive, or non-positive values.
    Guarantees strict cheap < expensive.
    """
    raw = [
        max(0.10, float(too_cheap)),
        max(0.10, float(cheap)),
        max(0.10, float(expensive)),
        max(0.10, float(too_expensive)),
    ]

    # If already monotonic with strict cheap < expensive:
    if raw[0] <= raw[1] and raw[1] < raw[2] and raw[2] <= raw[3]:
        return round(raw[0], 2), round(raw[1], 2), round(raw[2], 2), round(raw[3], 2)

    # Sort values to establish base order
    s = sorted(raw)
    tc, c, exp, te = s[0], s[1], s[2], s[3]

    # Enforce strict inequality between cheap and expensive
    if exp <= c:
        exp = round(c + min_step, 2)
        if te < exp:
            te = exp

    if tc > c:
        tc = c

    return round(tc, 2), round(c, 2), round(exp, 2), round(te, 2)


# ---------------------------------------------------------------------------
# Structured JSON Schemas for OpenAI (one per part of the two-part interview)
# ---------------------------------------------------------------------------
# Part 1: the four Van Westendorp price limits, asked before the persona sees the price
PRICE_LIMITS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "persona_id": {"type": "string"},
        "too_cheap": {"type": "number"},
        "cheap": {"type": "number"},
        "expensive": {"type": "number"},
        "too_expensive": {"type": "number"},
    },
    "required": ["persona_id", "too_cheap", "cheap", "expensive", "too_expensive"],
    "additionalProperties": False,
}

# Part 2: the purchase decision at the seller's price, asked after the price is revealed
PURCHASE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "persona_id": {"type": "string"},
        "buy_probability_pct": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "sentiment_label": {
            "type": "string",
            "enum": ["Positive", "Neutral", "Negative"],
        },
        "in_character_quote": {"type": "string"},
        "primary_objection": {
            "type": "string",
            "maxLength": 120,
        },
    },
    "required": [
        "persona_id",
        "buy_probability_pct",
        "sentiment_label",
        "in_character_quote",
        "primary_objection",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an advanced consumer simulation agent representing an authentic Singapore resident in an empirical market research focus group.
You MUST strictly adopt the persona's demographic background, dwelling and household income. The eating-out figure in your profile is an average for your home type, a guide to typical spending, not a hard limit.
You are evaluating a proposed business concept planned for your neighborhood.
You will be interviewed in two parts: first, before you are told the seller's price, you give your own 4 Van Westendorp price thresholds (in SGD); then you are told the price and say how likely you are to buy (0-100%).
Your responses must be grounded in Singapore's local consumer reality (hawker vs cafe price gaps, CDC voucher habits, neighborhood foot traffic).
Ground rules:
- Do not invent other facts about yourself (no job, children, hobbies or health unless stated in your profile).
- The monthly household income is market income for the whole household, combined, not just you: income from work plus interest, dividends, rent, pensions, CPF payouts and allowances from family living elsewhere. It excludes government support (such as Silver Support or CDC vouchers) and money drawn from savings.
- If the price is too high for your household, or the product does not suit you, say you would not buy. Answer honestly; most real people say no to most new products.
You must output a single JSON object matching the requested schema strictly. Do not include markdown code fences or extraneous text."""

PRICE_HIDDEN = "[price not shown]"

# Money amounts in a pitch: S$7, $6.50, SGD 12, 7 dollars, 4.5 bucks, 8sgd
_PRICE_PATTERN = re.compile(
    r"(?:S\$|SGD\s*|\$)\s*\d+(?:[.,]\d+)?"
    r"|\b\d+(?:[.,]\d+)?\s*(?:sgd|dollars?|bucks)\b",
    re.IGNORECASE,
)


def hide_prices(text: str) -> str:
    """Replaces money amounts in pitch text, so part 1 of the interview never shows the seller's price."""
    return _PRICE_PATTERN.sub(PRICE_HIDDEN, text or "")


def format_rag_context(rag_chunks: Optional[List[Any]], planning_area: str) -> str:
    """Formats Layer 2 ChromaDB chunks into a structured neighborhood context block."""
    if not rag_chunks:
        return f"Standard neighborhood market dynamics for {planning_area}."

    lines = [f"Local Neighborhood Context & Market Conditions in {planning_area}:"]
    category_labels = {
        "pricing_benchmarks": "Local Pricing Norms & Benchmarks",
        "competitors_and_supply": "Nearby Competitors & Food Options",
        "foot_traffic_and_lifestyle": "Neighborhood Lifestyle & Foot Traffic",
    }
    for chunk in rag_chunks:
        cat = getattr(chunk, "category", "") or (chunk.get("category", "") if isinstance(chunk, dict) else "")
        text = getattr(chunk, "text", "") or (chunk.get("text", "") if isinstance(chunk, dict) else str(chunk))
        label = category_labels.get(cat, cat.replace("_", " ").title() if cat else "Local Context")
        lines.append(f"- [{label}]: {text}")
    return "\n".join(lines)


def senior_profile_lines(persona: SyntheticPersona) -> str:
    """
    Extra profile lines for older personas (empty for anyone under 60):
      60+: labour force status (2025 national rate by age and sex)
      65+: living arrangement (2025 national shares by age)
    """
    lines = []
    in_labour_force = getattr(persona, "in_labour_force", None)
    if in_labour_force is not None:
        lines.append(
            "Work: in the labour force (working or looking for work)"
            if in_labour_force
            else "Work: not in the labour force (not working and not looking for work)"
        )
    arrangement = living_arrangement_text(getattr(persona, "living_arrangement", None))
    if arrangement:
        lines.append(f"Living arrangement: {arrangement}")
    return "".join(f"{line}\n" for line in lines)


def build_price_limits_prompt(
    persona: SyntheticPersona,
    concept: DecomposedConcept,
    rag_chunks: Optional[List[Any]] = None,
) -> str:
    """
    Part 1 of the interview: persona profile, the concept with every price removed, and local context.
    Asks only for the 4 price thresholds, so they are not anchored on the seller's price.
    Local prices in the neighbourhood context are kept: they are reference points, not the seller's price.
    """
    description = (
        getattr(concept, "concept_description", None)
        or getattr(concept, "description", "")
        or "No description provided."
    )
    rag_text = format_rag_context(rag_chunks, concept.target_planning_area)

    return f"""--- PERSONA PROFILE ({persona.persona_id}) ---
Age: {persona.age} ({persona.age_bracket})
Gender: {persona.gender}
Dwelling: {persona.dwelling_type}
{senior_profile_lines(persona)}Monthly Household Market Income: {persona.income_bracket} (whole household combined; work, rent, pensions, CPF payouts and family allowances; excludes government support)
Typical Eating-Out Spend: S${persona.daily_dining_budget_sgd:.2f} per person per day (HES 2023 average for your home type, adjusted to 2025 prices; not your personal limit)

--- PROPOSED BUSINESS CONCEPT ---
Concept Name: {hide_prices(concept.concept_name)}
Category: {concept.category_type}
Description: {hide_prices(description)}
Target Planning Area: {concept.target_planning_area}

--- LOCAL NEIGHBORHOOD CONTEXT (RAG) ---
{rag_text}

--- YOUR TASK (PART 1 OF 2: YOUR OWN PRICES) ---
Adopt this persona completely. The seller has not told you their price yet. Consider your household income, the typical eating-out spend of about S${persona.daily_dining_budget_sgd:.2f} per day for households like yours, and the local prices and alternatives in {concept.target_planning_area} described in the neighbourhood context above.
Answer the following in character, from your own sense of value:
1. `persona_id`: "{persona.persona_id}"
2. `too_cheap`: Price in SGD below which you would question the product's quality or freshness.
3. `cheap`: Price in SGD you would consider a bargain / great value for money.
4. `expensive`: Price in SGD that feels pricey, but you might still consider buying as a treat.
5. `too_expensive`: Price in SGD where you would completely refuse to purchase because it is unreasonably expensive.
(Note: your thresholds should logically satisfy: too_cheap <= cheap < expensive <= too_expensive).
"""


def build_purchase_prompt(persona: SyntheticPersona, concept: DecomposedConcept) -> str:
    """
    Part 2 of the interview, sent in the same conversation after the persona's part 1 answer:
    reveals the seller's price and asks for the purchase decision.
    """
    return f"""--- YOUR TASK (PART 2 OF 2: THE SELLER'S PRICE) ---
The seller's price for {concept.concept_name} is S${concept.candidate_price_sgd:.2f}.
Stay in character as persona {persona.persona_id}, and keep your own price answers from part 1 in mind.
Answer the following:
1. `persona_id`: "{persona.persona_id}"
2. `buy_probability_pct`: An integer from 0 to 100 representing how likely you are to buy this product at the seller's price of S${concept.candidate_price_sgd:.2f} during a typical week.
3. `sentiment_label`: "Positive", "Neutral", or "Negative" reflecting your general reaction to this business concept at this price.
4. `in_character_quote`: A short 1-2 sentence authentic reaction quote, written in standard Singapore English.
5. `primary_objection`: Your main hesitation, criticism, or constraint in 120 characters or fewer (e.g. "Price too high for daily kopi", "Prefer traditional hawker stall", "No CDC vouchers accepted").
"""


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Fallback Engine
# ---------------------------------------------------------------------------
_CATEGORY_ARCHETYPE_ANCHORS: Dict[str, Dict[str, float]] = {
    "beverage": {
        "retiree": 1.50,
        "working_class": 2.20,
        "student": 3.80,
        "heartland_family": 3.20,
        "affluent_pro": 6.80,
    },
    "bakery": {
        "retiree": 2.00,
        "working_class": 2.60,
        "student": 3.80,
        "heartland_family": 4.20,
        "affluent_pro": 7.50,
    },
    "full_service_dining": {
        "retiree": 6.00,
        "working_class": 8.50,
        "student": 13.00,
        "heartland_family": 16.00,
        "affluent_pro": 28.00,
    },
    "retail": {
        "retiree": 5.00,
        "working_class": 8.00,
        "student": 12.00,
        "heartland_family": 18.00,
        "affluent_pro": 35.00,
    },
}

_DWELLING_MULTIPLIERS: Dict[str, float] = {
    "1-2 room": 0.80,
    "3 room": 0.90,
    "4 room": 1.00,
    "5 room/exec": 1.15,
    "condo": 1.45,
    "landed": 1.65,
}


def evaluate_persona_rule_based(
    persona: SyntheticPersona,
    concept: DecomposedConcept,
    rag_chunks: Optional[List[Any]] = None,
) -> PersonaEvaluationOutput:
    """
    Deterministic rule-based persona evaluation.
    Grounded in persona demographics, HES spending benchmarks, and category anchors.
    Ensures that test suites, offline runs, and dry-runs never crash.
    """
    cat = getattr(concept, "category_type", "beverage")
    cat_anchors = _CATEGORY_ARCHETYPE_ANCHORS.get(cat, _CATEGORY_ARCHETYPE_ANCHORS["beverage"])
    arch = getattr(persona, "archetype", "heartland_family")
    base_anchor = cat_anchors.get(arch, 3.50)

    # Adjust anchor by dwelling and daily eating-out spend.
    # Budget is compared with the average of the six HES dwelling figures (~S$10.99/day),
    # so an average household scores 1.0 (census range ~0.71 for 1-2 room to ~1.37 for condo).
    dwelling = getattr(persona, "dwelling_type", "4 room")
    dwell_mult = _DWELLING_MULTIPLIERS.get(dwelling, 1.0)
    reference_budget = average_daily_eating_out_sgd()
    budget = getattr(persona, "daily_dining_budget_sgd", reference_budget)
    budget_mult = min(1.6, max(0.6, budget / reference_budget))

    anchor_price = base_anchor * (dwell_mult * 0.65 + budget_mult * 0.35)

    # Van Westendorp thresholds
    raw_tc = max(0.60, round(anchor_price * 0.45, 2))
    raw_c = max(raw_tc + 0.20, round(anchor_price * 0.75, 2))
    raw_exp = max(raw_c + 0.30, round(anchor_price * 1.30, 2))
    raw_te = max(raw_exp + 0.50, round(anchor_price * 1.80, 2))

    tc, c, exp, te = repair_price_monotonicity(raw_tc, raw_c, raw_exp, raw_te)

    # Evaluate candidate price
    cand_p = getattr(concept, "candidate_price_sgd", 4.50)
    if cand_p <= c:
        prob = 75.0 + 20.0 * max(0.0, 1.0 - (cand_p / max(0.1, c)))
    elif cand_p <= exp:
        prob = 35.0 + 40.0 * ((exp - cand_p) / max(0.1, exp - c))
    elif cand_p <= te:
        prob = 10.0 + 25.0 * ((te - cand_p) / max(0.1, te - exp))
    else:
        prob = max(2.0, 10.0 - (cand_p - te) * 2.0)

    buy_prob = int(round(max(0.0, min(100.0, prob))))

    # Sentiment classification
    if buy_prob >= 60:
        sentiment: Literal["Positive", "Neutral", "Negative"] = "Positive"
    elif buy_prob >= 35:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"

    # Quote & primary objection in standard Singapore English, chosen by sentiment
    if sentiment == "Positive":
        quote = f"At S${cand_p:.2f}, this represents good value for money and fits nicely into my weekly budget."
        objection = "None, highly competitive pricing."
    elif sentiment == "Neutral":
        quote = f"The S${cand_p:.2f} price point is acceptable, though whether I return depends on actual quality."
        objection = "Price is marginally high for frequent repeat visits."
    else:
        quote = f"S${cand_p:.2f} is difficult to justify when comparable quality options exist nearby at a lower price."
        objection = "Poor value proposition compared to existing alternatives."

    return PersonaEvaluationOutput(
        persona_id=persona.persona_id,
        buy_probability_pct=buy_prob,
        too_cheap=tc,
        cheap=c,
        expensive=exp,
        too_expensive=te,
        sentiment_label=sentiment,
        in_character_quote=quote,
        primary_objection=objection[:120],
    )


# ---------------------------------------------------------------------------
# LLM Async Invocation Worker (two-part interview per persona)
# ---------------------------------------------------------------------------
async def _ask_json(
    client: Any,
    model: Optional[str],
    messages: List[Dict[str, str]],
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """One AI call that must return a JSON object: strict json_schema first, json_object as a fallback."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            temperature=0.7,
        )
    except Exception as schema_err:
        logger.debug(f"json_schema mode failed ({schema_err}); trying json_object.")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )

    raw_text = (response.choices[0].message.content or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text)
    return json.loads(raw_text)


async def _evaluate_single_persona(
    sem: asyncio.Semaphore,
    client: Optional[Any],
    persona: SyntheticPersona,
    concept: DecomposedConcept,
    rag_chunks: Optional[List[Any]],
    use_mock: bool,
    model: Optional[str] = None,
) -> PersonaEvaluationOutput:
    """
    Evaluates a single persona with bounded concurrency and fallback, in two AI calls:
      1. price limits: the concept without its price -> too_cheap / cheap / expensive / too_expensive
      2. purchase: same conversation, price revealed -> buy %, sentiment, quote, objection
    If either call fails, the persona falls back to the rule-based evaluation.
    """
    if use_mock or client is None:
        return evaluate_persona_rule_based(persona, concept, rag_chunks)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_price_limits_prompt(persona, concept, rag_chunks)},
    ]

    async with sem:
        try:
            # Part 1: price limits, before the seller's price is shown
            limits = await _ask_json(client, model, messages, "persona_price_limits", PRICE_LIMITS_SCHEMA)
            tc, c, exp, te = repair_price_monotonicity(
                limits.get("too_cheap", 1.0),
                limits.get("cheap", 2.0),
                limits.get("expensive", 4.0),
                limits.get("too_expensive", 6.0),
            )

            # Part 2: reveal the price; the persona's own (repaired) part 1 answer stays in the conversation.
            # A new list, so the part 1 request is left exactly as it was sent.
            part2_messages = messages + [
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "persona_id": persona.persona_id,
                        "too_cheap": tc, "cheap": c, "expensive": exp, "too_expensive": te,
                    }),
                },
                {"role": "user", "content": build_purchase_prompt(persona, concept)},
            ]
            decision = await _ask_json(client, model, part2_messages, "persona_purchase", PURCHASE_SCHEMA)

            result = PersonaEvaluationOutput(
                persona_id=persona.persona_id,
                buy_probability_pct=decision.get("buy_probability_pct", 50),
                too_cheap=tc,
                cheap=c,
                expensive=exp,
                too_expensive=te,
                sentiment_label=decision.get("sentiment_label", "Neutral"),
                in_character_quote=str(decision.get("in_character_quote", "")),
                primary_objection=str(decision.get("primary_objection", "None"))[:120],
            )
            result._used_ai = True
            return result
        except Exception as exc:
            logger.warning(
                f"LLM evaluation failed for persona {persona.persona_id} ({exc}); using rule-based fallback."
            )
            return evaluate_persona_rule_based(persona, concept, rag_chunks)


# ---------------------------------------------------------------------------
# Main Async Execution Function
# ---------------------------------------------------------------------------
async def run_persona_evaluations(
    personas: list[SyntheticPersona],
    concept: DecomposedConcept,
    rag_chunks: list[Any] = None,
    use_mock: bool = False,
    concurrency: int = 25,
) -> list[PersonaEvaluationOutput]:
    """
    Evaluates a cohort of synthetic personas against a candidate concept.
    Runs evaluations concurrently bounded by asyncio.Semaphore(concurrency) (default 25).

    Args:
        personas: List of 30 sampled SyntheticPersona profiles.
        concept: Decomposed candidate business concept.
        rag_chunks: Qualitative ChromaDB chunks for local context.
        use_mock: If True, bypasses LLM and returns deterministic rule-based results.

    Returns:
        list[PersonaEvaluationOutput]: Validated, monotonic evaluation results for all personas.
    """
    if not personas:
        return []

    client = None
    model: Optional[str] = None
    if not use_mock:
        settings = get_llm_settings()
        if settings:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(**settings.client_kwargs())
                model = settings.model
                logger.info(f"Persona engine: provider={settings.provider}, model={settings.model}")
            except Exception as e:
                logger.warning(f"Could not initialize AsyncOpenAI: {e}; falling back to mock.")
                client = None
        else:
            logger.info(f"Persona engine: {describe_missing_provider()}; running deterministic evaluation engine.")

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _evaluate_single_persona(
            sem=sem,
            client=client,
            persona=p,
            concept=concept,
            rag_chunks=rag_chunks,
            use_mock=use_mock,
            model=model,
        )
        for p in personas
    ]

    results = await asyncio.gather(*tasks)
    return list(results)
