"""
backend/app/agents/synthesis_agent.py
---------------------------------------
Macro Synthesis & Strategic Pivot Agent for SimSG:
Performs high-level strategic reasoning over synthetic cohort evaluations,
demographic cohort breakdowns, and Van Westendorp Price Sensitivity Meter (PSM)
econometric curves.

Core Capabilities:
  - Synthesizes findings using `gpt-4o-mini` with structured JSON output enforcement.
  - Generates clear strategic verdict: "VIABLE", "MODERATE RISK", or "HIGH RISK - PIVOT REQUIRED".
  - Clusters qualitative persona objections with observed cohort frequency and direct quotes.
  - Delivers actionable Singapore-specific operational pivots (pricing adjustment to OPP,
    menu bundling, CDC voucher enablement, off-peak concessions).
  - Estimates simulated post-pivot purchase intent uplift.
  - Resilient offline fallback: Deterministic rule-based synthesis when API key is
    missing, dry-run is requested, or API errors occur.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("simsg.synthesis_agent")

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
    from app.services.query_decomposer import DecomposedConcept
    from app.services.econometrics import EconometricAnalysisResult
    from app.agents.execution_engine import PersonaEvaluationOutput
except ImportError:
    try:
        from backend.app.services.query_decomposer import DecomposedConcept
        from backend.app.services.econometrics import EconometricAnalysisResult
        from backend.app.agents.execution_engine import PersonaEvaluationOutput
    except ImportError:
        DecomposedConcept = Any  # type: ignore
        EconometricAnalysisResult = Any  # type: ignore
        PersonaEvaluationOutput = Any  # type: ignore


# ---------------------------------------------------------------------------
# Pydantic Output Models
# ---------------------------------------------------------------------------
class ClusteredObjection(BaseModel):
    """
    A synthesized cluster of recurring persona objections.
    """

    theme: str = Field(
        ...,
        description="Core objection theme (e.g. 'Price Disparity vs Traditional Hawker Kopi')",
    )
    frequency_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated percentage of the cohort expressing this concern",
    )
    representative_quotes: List[str] = Field(
        ...,
        description="Direct quotes from evaluated personas illustrating this theme",
    )

    @field_validator("frequency_pct", mode="before")
    @classmethod
    def clamp_frequency(cls, v: Any) -> float:
        try:
            return round(max(0.0, min(100.0, float(v))), 1)
        except Exception:
            return 0.0


class MacroSynthesisReport(BaseModel):
    """
    Consolidated strategic verdict and executive recommendation report.
    """

    verdict_badge: Literal["VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"] = Field(
        ...,
        description="Overall business model viability verdict",
    )
    executive_summary: str = Field(
        ...,
        description="3-4 sentence strategic takeaway summarizing market fit and risk",
    )
    clustered_objections: List[ClusteredObjection] = Field(
        ...,
        description="Top recurring customer objections clustered by theme with frequency and quotes",
    )
    recommended_pivot: str = Field(
        ...,
        description="Concrete operational pivot (price adjustment to OPP, bundles, CDC vouchers)",
    )
    simulated_post_pivot_uplift: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated modeled purchase intent percentage after applying the pivot",
    )

    @field_validator("simulated_post_pivot_uplift", mode="before")
    @classmethod
    def clamp_uplift(cls, v: Any) -> float:
        try:
            return round(max(0.0, min(100.0, float(v))), 1)
        except Exception:
            return 50.0

    @field_validator("verdict_badge", mode="before")
    @classmethod
    def normalize_verdict(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if s in ("VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"):
            return s
        if "HIGH" in s or "PIVOT" in s:
            return "HIGH RISK - PIVOT REQUIRED"
        if "MODERATE" in s or "MEDIUM" in s:
            return "MODERATE RISK"
        return "VIABLE"


# ---------------------------------------------------------------------------
# Structured JSON Schema for OpenAI
# ---------------------------------------------------------------------------
MACRO_SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict_badge": {
            "type": "string",
            "enum": ["VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"],
        },
        "executive_summary": {"type": "string"},
        "clustered_objections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "frequency_pct": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    "representative_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["theme", "frequency_pct", "representative_quotes"],
                "additionalProperties": False,
            },
        },
        "recommended_pivot": {"type": "string"},
        "simulated_post_pivot_uplift": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 100.0,
        },
    },
    "required": [
        "verdict_badge",
        "executive_summary",
        "clustered_objections",
        "recommended_pivot",
        "simulated_post_pivot_uplift",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------
SYNTHESIS_SYSTEM_PROMPT = """You are the Chief Strategy Officer and Lead Econometrician for SimSG, Singapore's advanced consumer simulation platform.
You analyze empirical focus group data, Van Westendorp price sensitivity curves, and demographic cohort metrics to provide C-suite strategic guidance for Singapore SMEs and F&B businesses.
You must synthesize findings into an actionable, rigorous report with a verdict badge, 3-4 sentence executive summary, clustered objections with authentic representative quotes, concrete operational pivot recommendations (including pricing, bundling, and CDC voucher integration), and simulated post-pivot purchase intent uplift.
Respond with a single JSON object strictly matching the requested schema. Do not output markdown fences or extraneous text."""


def build_synthesis_prompt(
    concept: DecomposedConcept,
    evaluations: List[PersonaEvaluationOutput],
    econometrics: EconometricAnalysisResult,
) -> str:
    """Constructs the prompt for the macro synthesis LLM call."""
    concept_name = getattr(concept, "concept_name", "Concept")
    area = getattr(concept, "target_planning_area", "Singapore")
    cand_price = getattr(concept, "candidate_price_sgd", 4.50)
    category = getattr(concept, "category_type", "beverage")
    desc = (
        getattr(concept, "concept_description", None)
        or getattr(concept, "description", "")
        or ""
    )

    # Econometric indicators
    mean_intent = getattr(econometrics, "mean_buy_intent_pct", 0.0)
    net_sentiment = getattr(econometrics, "net_sentiment_score", 0.0)
    sentiment_breakdown = getattr(econometrics, "sentiment_breakdown", {})
    psm = getattr(econometrics, "psm_intersections", None)
    pmc = getattr(psm, "pmc", 0.0) if psm else 0.0
    opp = getattr(psm, "opp", 0.0) if psm else 0.0
    ipp = getattr(psm, "ipp", 0.0) if psm else 0.0
    pme = getattr(psm, "pme", 0.0) if psm else 0.0

    # Cohort breakdown formatting
    cohort_lines = []
    cohort_list = getattr(econometrics, "cohort_breakdown", [])
    for c in cohort_list:
        c_name = getattr(c, "cohort_name", "Cohort")
        c_intent = getattr(c, "mean_buy_intent_pct", 0.0)
        c_ceiling = getattr(c, "median_price_ceiling_sgd", 0.0)
        c_share = getattr(c, "population_share_pct", 0.0)
        cohort_lines.append(
            f"  - {c_name} (Share: {c_share:.1f}%): Mean Buy Intent {c_intent:.1f}%, Median Price Ceiling S${c_ceiling:.2f}"
        )
    cohort_text = "\n".join(cohort_lines) if cohort_lines else "  No cohort breakdown available."

    # Persona qualitative quotes and objections
    sample_feedback_lines = []
    for e in evaluations[:15]:  # include representative sample to manage context window
        quote = getattr(e, "in_character_quote", "")
        obj = getattr(e, "primary_objection", "")
        sentiment = getattr(e, "sentiment_label", "")
        pid = getattr(e, "persona_id", "")
        sample_feedback_lines.append(
            f"  - [{pid}] ({sentiment}): \"{quote}\" (Objection: {obj})"
        )
    feedback_text = "\n".join(sample_feedback_lines)

    return f"""--- PROPOSED CONCEPT ---
Concept Name: {concept_name}
Target Planning Area: {area}
Candidate Price: S${cand_price:.2f}
Category: {category}
Description: {desc}

--- QUANTITATIVE ECONOMETRIC RESULTS ---
- Mean Purchase Intent: {mean_intent:.1f}%
- Net Sentiment Score: {net_sentiment:+.2f} (from -1.0 to +1.0)
- Sentiment Breakdown: {sentiment_breakdown}
- Van Westendorp Price Sensitivity Meter (PSM) Intersections:
  * Point of Marginal Cheapness (PMC): S${pmc:.2f}
  * Optimal Price Point (OPP): S${opp:.2f}
  * Indifference Price Point (IPP): S${ipp:.2f}
  * Point of Marginal Expensiveness (PME): S${pme:.2f}

--- COHORT SEGMENTATION ANALYSIS ---
{cohort_text}

--- QUALITATIVE FOCUS GROUP FEEDBACK (30 PERSONAS) ---
{feedback_text}

--- YOUR TASK ---
Perform a comprehensive strategic synthesis:
1. `verdict_badge`: Exactly one of "VIABLE", "MODERATE RISK", or "HIGH RISK - PIVOT REQUIRED".
   - VIABLE: Candidate price is within OPP to IPP range, mean buy intent >= 55%, positive sentiment.
   - MODERATE RISK: Candidate price is between IPP and PME, or intent between 35-55%, with mixed resistance.
   - HIGH RISK - PIVOT REQUIRED: Candidate price exceeds PME (S${pme:.2f}), or intent < 35%, or substantial price rejection from heartland cohorts.
2. `executive_summary`: 3-4 sentence strategic takeaway analyzing market viability, price elasticity relative to OPP (S${opp:.2f}), and cohort dynamics in {area}.
3. `clustered_objections`: Cluster recurring concerns into 2-4 themes. Each cluster must have `theme`, `frequency_pct` (0-100), and 2-3 `representative_quotes`.
4. `recommended_pivot`: Concrete operational pivot for the Singapore market. Include price repositioning towards OPP, value bundling, CDC voucher acceptance, and daypart concessions.
5. `simulated_post_pivot_uplift`: Modeled realistic purchase intent percentage (0-100) after executing the pivot.
"""


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Fallback Generator
# ---------------------------------------------------------------------------
def generate_macro_synthesis_rule_based(
    concept: DecomposedConcept,
    evaluations: List[PersonaEvaluationOutput],
    econometrics: EconometricAnalysisResult,
) -> MacroSynthesisReport:
    """
    Deterministic rule-based strategic synthesis generator.
    Guarantees that automated tests and offline dry-runs never crash.
    """
    concept_name = getattr(concept, "concept_name", "Concept")
    area = getattr(concept, "target_planning_area", "Ang Mo Kio")
    cand_price = getattr(concept, "candidate_price_sgd", 4.50)

    mean_intent = getattr(econometrics, "mean_buy_intent_pct", 38.0)
    net_sentiment = getattr(econometrics, "net_sentiment_score", -0.05)
    psm = getattr(econometrics, "psm_intersections", None)
    opp = getattr(psm, "opp", 3.20) if psm else 3.20
    pme = getattr(psm, "pme", 5.20) if psm else 5.20

    # Determine Verdict
    if cand_price > pme or mean_intent < 35.0 or net_sentiment < -0.15:
        verdict: Literal["VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"] = "HIGH RISK - PIVOT REQUIRED"
    elif mean_intent >= 55.0 and net_sentiment >= 0.20 and cand_price <= pme:
        verdict = "VIABLE"
    else:
        verdict = "MODERATE RISK"

    # Executive Summary (3-4 sentences)
    price_assessment = (
        f"exceeds the Point of Marginal Expensiveness (PME: S${pme:.2f}), triggering severe consumer resistance"
        if cand_price > pme
        else f"sits above the Optimal Price Point (OPP: S${opp:.2f}), compressing broad heartland appeal"
        if cand_price > opp
        else f"aligns favorably with the Optimal Price Point (OPP: S${opp:.2f})"
    )

    s1 = f"The proposed concept '{concept_name}' in {area} at S${cand_price:.2f} receives a '{verdict}' viability verdict based on a 30-persona simulated cohort."
    s2 = f"Van Westendorp econometric analysis reveals an empirical Optimal Price Point of S${opp:.2f} and a Point of Marginal Expensiveness of S${pme:.2f}; current pricing {price_assessment}."
    s3 = f"While affluent private property dwellers and young adults express moderate interest, working-class and retiree cohorts exhibit substantial price sensitivity, anchoring heavily to traditional hawker alternatives."
    s4 = f"Executing a strategic pivot focused on price repositioning towards S${opp:.2f} and introducing value bundling will be essential to establish sustainable foot traffic and margin."
    executive_summary = f"{s1} {s2} {s3} {s4}"

    # Cluster Objections from actual evaluation quotes
    quotes_hawker: List[str] = []
    quotes_budget: List[str] = []
    quotes_value: List[str] = []

    for e in evaluations:
        q = getattr(e, "in_character_quote", "")
        obj = getattr(e, "primary_objection", "")
        text = f"{q} {obj}".lower()
        if any(w in text for w in ("hawker", "kopi", "coffeeshop", "ex", "expensive", "siao", "price")):
            quotes_hawker.append(q)
        elif any(w in text for w in ("budget", "wallet", "daily", "everyday", "treat", "frequent")):
            quotes_budget.append(q)
        else:
            quotes_value.append(q)

    n_evals = max(1, len(evaluations))
    c1_count = len(quotes_hawker)
    c2_count = len(quotes_budget)
    c3_count = len(quotes_value)

    clustered = [
        ClusteredObjection(
            theme="Price Disparity vs Traditional Hawker Alternatives",
            frequency_pct=round((c1_count / n_evals) * 100, 1),
            representative_quotes=quotes_hawker[:3] or [
                f"S${cand_price:.2f} is too expensive compared to hawker alternatives."
            ],
        ),
        ClusteredObjection(
            theme="Daily Dining Budget & Habit Constraints",
            frequency_pct=round((c2_count / n_evals) * 100, 1),
            representative_quotes=quotes_budget[:3] or [
                "Good for occasional treats, but cannot sustain as a daily routine meal."
            ],
        ),
    ]

    if c3_count > 0:
        clustered.append(
            ClusteredObjection(
                theme="Value Perception & Proposition Clarity",
                frequency_pct=round((c3_count / n_evals) * 100, 1),
                representative_quotes=quotes_value[:3] or [
                    "Willing to try once, but ongoing visits require clearer value or bundle options."
                ],
            )
        )

    # Recommended Pivot
    recommended_pivot = (
        f"1. Reposition core entry product from S${cand_price:.2f} down towards the Optimal Price Point of S${opp:.2f}.\n"
        f"2. Introduce a two-tier menu architecture: Standard offering at S${opp:.2f} alongside premium/artisanal selections at S${cand_price:.2f}.\n"
        f"3. Enable Community Development Council (CDC) voucher acceptance to eliminate friction for heartland residents in {area}.\n"
        f"4. Launch off-peak bundle deals (e.g. 2:00 PM – 5:00 PM tea-time combos) to drive student and retiree foot traffic."
    )

    # Simulated post-pivot uplift
    simulated_uplift = round(min(88.0, max(mean_intent + 22.0, 58.0)), 1)

    return MacroSynthesisReport(
        verdict_badge=verdict,
        executive_summary=executive_summary,
        clustered_objections=clustered,
        recommended_pivot=recommended_pivot,
        simulated_post_pivot_uplift=simulated_uplift,
    )


# ---------------------------------------------------------------------------
# Main Async Synthesis Function
# ---------------------------------------------------------------------------
async def generate_macro_synthesis(
    concept: DecomposedConcept,
    evaluations: list[PersonaEvaluationOutput],
    econometrics: EconometricAnalysisResult,
    use_mock: bool = False,
) -> MacroSynthesisReport:
    """
    Generates the macro synthesis report and strategic pivot recommendations.
    Uses `gpt-4o-mini` with structured JSON enforcement, falling back gracefully
    to deterministic rule-based synthesis if credentials are absent or errors occur.

    Args:
        concept: Decomposed candidate business concept.
        evaluations: List of PersonaEvaluationOutput objects from all 30 personas.
        econometrics: Quantitative report from EconometricsService.
        use_mock: If True, bypasses LLM and returns deterministic rule-based report.

    Returns:
        MacroSynthesisReport: Complete strategic verdict, summary, clusters, and pivot.
    """
    if use_mock:
        return generate_macro_synthesis_rule_based(concept, evaluations, econometrics)

    api_key = get_openai_api_key()
    if not api_key:
        logger.info("No OpenAI API key found; generating deterministic macro synthesis report.")
        return generate_macro_synthesis_rule_based(concept, evaluations, econometrics)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        prompt = build_synthesis_prompt(concept, evaluations, econometrics)
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Call gpt-4o-mini with structured output
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "macro_synthesis_report",
                        "strict": True,
                        "schema": MACRO_SYNTHESIS_SCHEMA,
                    },
                },
                temperature=0.6,
            )
        except Exception as schema_err:
            logger.debug(f"Synthesis json_schema failed ({schema_err}); trying json_object.")
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.6,
            )

        raw_text = response.choices[0].message.content or ""
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text)

        data = json.loads(raw_text)
        return MacroSynthesisReport(**data)

    except Exception as exc:
        logger.warning(
            f"LLM macro synthesis generation failed ({exc}); falling back to deterministic synthesis."
        )
        return generate_macro_synthesis_rule_based(concept, evaluations, econometrics)
