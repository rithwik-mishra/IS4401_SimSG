"""
backend/app/agents/execution_engine.py
----------------------------------------
Async Batch Persona Execution Engine for SimSG:
Evaluates synthetic cohorts of Singapore residents (30 personas) against a
decomposed business pitch and Layer 2 ChromaDB qualitative neighborhood context.

Core Capabilities:
  - Bounded concurrent execution using asyncio.gather and asyncio.Semaphore(10).
  - Uses `gpt-4o-mini` with structured JSON output enforcement.
  - Contextual prompt conditioning across demographic attributes, daily dining budget,
    price baselines, linguistic register, and local neighborhood conditions.
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
from pydantic import BaseModel, Field, field_validator

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


def get_openai_api_key(passed_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves OpenAI API key with explicit priority:
    1. passed_key
    2. OPEN_API_KEY from backend/.env
    3. OPENAI_API_KEY from backend/.env
    """
    key = passed_key or os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        key = key.strip()
        if key.startswith("your_") or key == "":
            return None
        return key
    return None


# ---------------------------------------------------------------------------
# Schema Imports (Robust across package relative and absolute paths)
# ---------------------------------------------------------------------------
try:
    from app.services.sampler import SyntheticPersona
    from app.services.query_decomposer import DecomposedConcept
except ImportError:
    try:
        from backend.app.services.sampler import SyntheticPersona
        from backend.app.services.query_decomposer import DecomposedConcept
    except ImportError:
        SyntheticPersona = Any  # type: ignore
        DecomposedConcept = Any  # type: ignore


# ---------------------------------------------------------------------------
# Pydantic Output Model
# ---------------------------------------------------------------------------
class PersonaEvaluationOutput(BaseModel):
    """
    Individual synthetic resident evaluation output for a business concept pitch.
    """

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
        description="Authentic reaction quote matching persona linguistic style",
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
# Structured JSON Schema for OpenAI
# ---------------------------------------------------------------------------
PERSONA_EVALUATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "persona_id": {"type": "string"},
        "buy_probability_pct": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "too_cheap": {"type": "number"},
        "cheap": {"type": "number"},
        "expensive": {"type": "number"},
        "too_expensive": {"type": "number"},
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
        "too_cheap",
        "cheap",
        "expensive",
        "too_expensive",
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
You MUST strictly adopt the persona's demographic background, dwelling, household income, daily dining budget, reference price anchors, and linguistic style.
You are evaluating a proposed business concept planned for your neighborhood.
Evaluate whether you would realistically buy this offering at the proposed candidate price, how likely you are to purchase (0-100%), and provide your 4 Van Westendorp price thresholds (in SGD).
Your responses must be grounded in Singapore's local consumer reality (hawker vs cafe price gaps, CDC voucher habits, neighborhood foot traffic).
You must output a single JSON object matching the requested schema strictly. Do not include markdown code fences or extraneous text."""


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


def build_persona_prompt(
    persona: SyntheticPersona,
    concept: DecomposedConcept,
    rag_chunks: Optional[List[Any]] = None,
) -> str:
    """Builds the comprehensive evaluation prompt for a single synthetic resident."""
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
Monthly Household Income: S${persona.monthly_household_income_sgd:,.0f} ({persona.income_bracket})
Consumer Archetype: {persona.archetype}
Daily Dining Budget: S${persona.daily_dining_budget_sgd:.2f}
Reference Baselines: Traditional Hawker Kopi S${persona.hawker_baseline_kopi_sgd:.2f} | Specialty Cafe Coffee S${persona.cafe_baseline_coffee_sgd:.2f}
Primary Home Language: {persona.home_language}
Linguistic Style: {persona.linguistic_style}

--- PROPOSED BUSINESS CONCEPT ---
Concept Name: {concept.concept_name}
Category: {concept.category_type}
Candidate Price: S${concept.candidate_price_sgd:.2f}
Description: {description}
Target Planning Area: {concept.target_planning_area}

--- LOCAL NEIGHBORHOOD CONTEXT (RAG) ---
{rag_text}

--- YOUR TASK ---
Adopt this persona completely. Consider your daily dining budget of S${persona.daily_dining_budget_sgd:.2f}, your reference price baselines, and local alternatives in {concept.target_planning_area}.
Answer the following in character:
1. `persona_id`: "{persona.persona_id}"
2. `buy_probability_pct`: An integer from 0 to 100 representing how likely you are to buy this product at the candidate price of S${concept.candidate_price_sgd:.2f} during a typical week.
3. `too_cheap`: Price in SGD below which you would question the product's quality or freshness.
4. `cheap`: Price in SGD you would consider a bargain / great value for money.
5. `expensive`: Price in SGD that feels pricey, but you might still consider buying as a treat.
6. `too_expensive`: Price in SGD where you would completely refuse to purchase because it is unreasonably expensive.
(Note: your thresholds should logically satisfy: too_cheap <= cheap < expensive <= too_expensive).
7. `sentiment_label`: "Positive", "Neutral", or "Negative" reflecting your general reaction to this business concept.
8. `in_character_quote`: A short 1-2 sentence authentic reaction quote spoken in your specified linguistic style ({persona.linguistic_style}).
9. `primary_objection`: Your main hesitation, criticism, or constraint in 120 characters or fewer (e.g. "Price too high for daily kopi", "Prefer traditional hawker stall", "No CDC vouchers accepted").
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

    # Adjust anchor by dwelling and daily budget
    dwelling = getattr(persona, "dwelling_type", "4 room")
    dwell_mult = _DWELLING_MULTIPLIERS.get(dwelling, 1.0)
    budget = getattr(persona, "daily_dining_budget_sgd", 25.0)
    budget_mult = min(1.6, max(0.6, budget / 25.0))

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

    # Authentic in-character quote & primary objection based on linguistic style
    ling_style = getattr(persona, "linguistic_style", "Mixed Code-Switch")
    kopi_base = getattr(persona, "hawker_baseline_kopi_sgd", 1.30)
    concept_name = getattr(concept, "concept_name", "this concept")

    if ling_style == "Colloquial Singlish":
        if sentiment == "Positive":
            quote = f"S${cand_p:.2f} steady lah! If standard is good, will definitely come down support one."
            objection = f"None, price is fair for {concept_name}."
        elif sentiment == "Neutral":
            quote = f"Can try once in a while lor, but every day spend S${cand_p:.2f} my wallet cannot take it lah."
            objection = "Price is a bit steep for daily breakfast or routine dining."
        else:
            quote = f"Wah lau S${cand_p:.2f} so ex! Better go downstairs coffeeshop drink kopi S${kopi_base:.2f} more shiok."
            objection = "Too expensive compared to traditional hawker alternatives."
    elif ling_style == "Mixed Code-Switch":
        if sentiment == "Positive":
            quote = f"S${cand_p:.2f} is actually quite reasonable for {concept_name}, I will definitely drop by to try."
            objection = f"None, reasonable price for {concept_name}."
        elif sentiment == "Neutral":
            quote = f"Pricing is okay-lah, maybe buy on weekends as a treat but not for everyday dining."
            objection = "Price slightly high for everyday routine meals."
        else:
            quote = f"S${cand_p:.2f} is a bit steep lah, better options available around the neighbourhood."
            objection = "High price barrier relative to neighborhood food options."
    else:  # Standard Singapore English
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
# LLM Async Invocation Worker
# ---------------------------------------------------------------------------
async def _evaluate_single_persona(
    sem: asyncio.Semaphore,
    client: Optional[Any],
    persona: SyntheticPersona,
    concept: DecomposedConcept,
    rag_chunks: Optional[List[Any]],
    use_mock: bool,
) -> PersonaEvaluationOutput:
    """Evaluates a single persona with bounded concurrency and fallback."""
    if use_mock or client is None:
        return evaluate_persona_rule_based(persona, concept, rag_chunks)

    prompt = build_persona_prompt(persona, concept, rag_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async with sem:
        try:
            # Attempt structured output via json_schema
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "persona_evaluation",
                            "strict": True,
                            "schema": PERSONA_EVALUATION_SCHEMA,
                        },
                    },
                    temperature=0.7,
                )
            except Exception as schema_err:
                logger.debug(f"json_schema mode failed ({schema_err}); trying json_object.")
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )

            raw_text = response.choices[0].message.content or ""
            # Strip potential fences
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text)

            data = json.loads(raw_text)

            # Monotonicity repair on price thresholds
            tc, c, exp, te = repair_price_monotonicity(
                data.get("too_cheap", 1.0),
                data.get("cheap", 2.0),
                data.get("expensive", 4.0),
                data.get("too_expensive", 6.0),
            )

            return PersonaEvaluationOutput(
                persona_id=persona.persona_id,
                buy_probability_pct=data.get("buy_probability_pct", 50),
                too_cheap=tc,
                cheap=c,
                expensive=exp,
                too_expensive=te,
                sentiment_label=data.get("sentiment_label", "Neutral"),
                in_character_quote=str(data.get("in_character_quote", "")),
                primary_objection=str(data.get("primary_objection", "None"))[:120],
            )
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
) -> list[PersonaEvaluationOutput]:
    """
    Evaluates a cohort of synthetic personas against a candidate concept.
    Runs evaluations concurrently bounded by asyncio.Semaphore(10).

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
    if not use_mock:
        api_key = get_openai_api_key()
        if api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not initialize AsyncOpenAI: {e}; falling back to mock.")
                client = None
        else:
            logger.info("No OpenAI API key found; running deterministic evaluation engine.")

    sem = asyncio.Semaphore(10)
    tasks = [
        _evaluate_single_persona(
            sem=sem,
            client=client,
            persona=p,
            concept=concept,
            rag_chunks=rag_chunks,
            use_mock=use_mock,
        )
        for p in personas
    ]

    results = await asyncio.gather(*tasks)
    return list(results)
