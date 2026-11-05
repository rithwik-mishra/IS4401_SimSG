"""
backend/app/services/sampler.py
---------------------------------
SimSG Synthetic Cohort Sampler:
Draws synthetic cohorts of Singapore adult residents from the census tables in
backend/data/sheets/processed/ (same logic as backend/agents/sampler.py).

Order of draws for each persona (each step uses only the tables named):
    1. dwelling + household income band   fitted_area_dwelling_income.csv (area)
    2. sex + 5-year age group (20+)       resident_area_age_sex.csv (area), independent of step 1
    3. exact age / income                 random within the drawn age group / income band
    4. daily eating-out spend             spend_per_member_dwelling.csv (HES 2023, by dwelling),
                                          looked up, not drawn: monthly spend per member / 30

The previous sampler (layer1_seed.json + hes_expenditure_benchmarks.json) is kept
in backend/app/legacy/sampler_layer1.py.

Usage (CLI):
    python backend/app/services/sampler.py "Ang Mo Kio" 30
    python -m backend.app.services.sampler "Tampines" 10 --seed 42

Usage (Python API):
    from backend.app.services.sampler import sample_synthetic_cohort
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=30, seed=42)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Census table paths
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent.parent  # backend/
_DATA_DIR = _BACKEND_DIR / "data"
CENSUS_DIR = _DATA_DIR / "sheets" / "processed"

FITTED_PATH = CENSUS_DIR / "fitted_area_dwelling_income.csv"
AGE_SEX_PATH = CENSUS_DIR / "resident_area_age_sex.csv"
SPEND_PATH = CENSUS_DIR / "spend_per_member_dwelling.csv"

MIN_AGE = 20  # personas are adults aged 20 and over
DAYS_PER_MONTH = 30
EATING_OUT_CATEGORY = "Food and Beverage Serving Services"  # HES: hawkers, food courts, cafes, restaurants

DwellingType = Literal["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]
DWELLING_TYPES: Tuple[str, ...] = ("1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SyntheticPersona(BaseModel):
    """
    Demographic and financial profile of one synthetic adult resident, drawn from
    the census tables (dwelling + income, sex + age) and HES 2023 eating-out spend.
    """

    persona_id: str = Field(..., description="Unique persona identifier (e.g. 'AMK-001')")
    planning_area: str = Field(..., description="Target Singapore planning area")
    age: int = Field(..., ge=20, description="Adult age (>= 20 years), random within the census age group")
    age_bracket: str = Field(..., description="Census 5-year age group, e.g. '35-39' or '90+'")
    gender: Literal["Male", "Female"] = Field(..., description="Census sex, drawn with age group by area")
    dwelling_type: DwellingType = Field(..., description="HDB or private dwelling category")
    monthly_household_income_sgd: float = Field(
        ..., ge=0.0, description="Monthly household income in SGD, random within the census income band"
    )
    income_bracket: str = Field(..., description="Census household income band, e.g. '$7,000 to $7,999'")
    archetype: Literal[
        "retiree", "student", "working_class", "heartland_family", "affluent_pro"
    ] = Field(..., description="SimSG consumer archetype (age, then home type); not shown to the persona AI")
    daily_dining_budget_sgd: float = Field(
        ...,
        ge=0.0,
        description=(
            "Average daily eating-out spend per household member for this dwelling type "
            "(HES 2023 monthly 'Food and Beverage Serving Services' / 30). An average, not a personal limit."
        ),
    )


class RepresentativePersona(BaseModel):
    """
    Flattened, table-ready record of a representative resident persona and their evaluation,
    portraying one of the 5 key Singapore demographic archetypes for the React dashboard table.
    """
    demographic_cohort: str = Field(
        ...,
        description="Demographic cohort: Senior / Retiree, Working-Class, Suburban Family, Young Adult / Student, or Affluent Private Property",
    )
    persona_id: str = Field(
        ...,
        description="Unique persona identifier (e.g. 'AMK-001')",
    )
    age: int = Field(
        ...,
        description="Age of the resident in years",
    )
    gender: str = Field(
        ...,
        description="Gender (Male / Female)",
    )
    dwelling_type: str = Field(
        ...,
        description="Housing / dwelling type (e.g. '1-2 room', '3 room', '4 room', '5 room/exec', 'condo', 'landed')",
    )
    monthly_income_sgd: float = Field(
        ...,
        description="Estimated monthly household income in SGD",
    )
    daily_dining_budget_sgd: float = Field(
        ...,
        description="Daily dining-out spend budget ceiling in SGD (SingStat HES 2023)",
    )
    buy_probability_pct: int = Field(
        ...,
        description="Simulated probability (0-100%) that this persona would purchase the offering",
    )
    sentiment: str = Field(
        ...,
        description="Sentiment classification: Positive, Neutral, or Negative",
    )
    price_ceiling_sgd: float = Field(
        ...,
        description="Maximum acceptable price ceiling in SGD before it is considered too expensive",
    )
    primary_objection: str = Field(
        ...,
        description="Core objection or hesitation voiced by this persona (max 120 chars)",
    )
    quote: str = Field(
        ...,
        description="In-character qualitative quote in authentic Singapore voice",
    )
    too_cheap: Optional[float] = Field(default=None, description="Too Cheap threshold in SGD")
    cheap: Optional[float] = Field(default=None, description="Cheap threshold in SGD")
    expensive: Optional[float] = Field(default=None, description="Expensive threshold in SGD")
    too_expensive: Optional[float] = Field(default=None, description="Too Expensive threshold in SGD")


# ---------------------------------------------------------------------------
# Census table loading (cached)
# ---------------------------------------------------------------------------
def _age_group_hi(label: str) -> Optional[int]:
    """'35-39' -> 39; open-ended '90+' -> None."""
    if label.endswith("+"):
        return None
    return int(label.split("-")[1])


def _income_label(lo: float, hi: Optional[float]) -> str:
    """Census income band as text: 'below $1,000', '$7,000 to $7,999', '$35,000 or more'."""
    if hi is None:
        return f"${int(lo):,} or more"
    if lo == 0:
        return f"below ${int(hi) + 1:,}"
    return f"${int(lo):,} to ${int(hi):,}"


def _fill_open_ended(rows: List[Dict[str, Any]], lo_key: str, hi_key: str) -> None:
    """
    Gives open-ended top bands ('90+', '$35,000+') the same width as the band below,
    e.g. 85-89 -> 90+ becomes 90-94; $30,000-34,999 -> $35,000+ becomes $35,000-39,999.
    """
    closed = [r for r in rows if r[hi_key] is not None]
    top = max(closed, key=lambda r: r[lo_key])
    width = top[hi_key] - top[lo_key]
    for r in rows:
        if r[hi_key] is None:
            r[hi_key] = r[lo_key] + width


@lru_cache(maxsize=1)
def _load_census_tables() -> Tuple[
    Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, float]
]:
    """
    Loads and caches the three census tables:
      households[area] -> rows of dwelling + income band, weighted by prob_in_area
      people[area]     -> rows of sex + age group (20+), weighted by residents
      eating_out_daily -> dwelling -> S$ eating-out spend per household member per day
    """
    for path in (FITTED_PATH, AGE_SEX_PATH, SPEND_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Census table not found at {path}")

    households: Dict[str, List[Dict[str, Any]]] = {}
    with FITTED_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            income_hi = float(row["income_hi"]) if row["income_hi"] else None
            lo = float(row["income_lo"])
            households.setdefault(row["area"], []).append({
                "dwelling": row["dwelling"],
                "income_lo": lo,
                "income_hi": income_hi,
                "income_label": _income_label(lo, income_hi),
                "weight": float(row["prob_in_area"]),
            })

    people: Dict[str, List[Dict[str, Any]]] = {}
    with AGE_SEX_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            age_lo = int(row["age_lo"])
            if age_lo < MIN_AGE:
                continue
            people.setdefault(row["area"], []).append({
                "age_group": row["age_group"],
                "age_lo": age_lo,
                "age_hi": _age_group_hi(row["age_group"]),
                "sex": row["sex"],
                "weight": float(row["residents"]),
            })

    eating_out_daily: Dict[str, float] = {}
    with SPEND_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] == EATING_OUT_CATEGORY:
                eating_out_daily[row["dwelling"]] = float(row["spend_per_member"]) / DAYS_PER_MONTH

    _fill_open_ended([r for rows in households.values() for r in rows], "income_lo", "income_hi")
    _fill_open_ended([r for rows in people.values() for r in rows], "age_lo", "age_hi")

    return households, people, eating_out_daily


def list_supported_areas() -> List[str]:
    """Planning areas that have both household and resident census tables."""
    households, people, _ = _load_census_tables()
    return sorted(set(households) & set(people))


def average_daily_eating_out_sgd() -> float:
    """
    Plain average of the six dwelling types' HES daily eating-out spend (about S$10.99).
    Used by the rule-based backup as its reference budget.
    """
    _, _, eating_out_daily = _load_census_tables()
    return sum(eating_out_daily[d] for d in DWELLING_TYPES) / len(DWELLING_TYPES)


# ---------------------------------------------------------------------------
# Persona IDs and archetypes
# ---------------------------------------------------------------------------
AREA_CODE_MAP: Dict[str, str] = {
    "ang mo kio": "AMK",
    "bukit merah": "BKM",
    "bukit timah": "BTM",
    "jurong east": "JRE",
    "outram": "OUT",
    "tampines": "TAM",
}


def _get_area_code(planning_area: str) -> str:
    """Returns a short 3-letter area code for persona ID generation."""
    clean = planning_area.lower().strip()
    if clean in AREA_CODE_MAP:
        return AREA_CODE_MAP[clean]
    words = planning_area.split()
    if len(words) > 1:
        return "".join(w[0] for w in words).upper()[:3]
    return planning_area[:3].upper()


# Each archetype maps to one demographic group shown in the app
ARCHETYPE_COHORT: Dict[str, str] = {
    "retiree": "Senior / Retiree",
    "student": "Young Adult / Student",
    "affluent_pro": "Affluent Private Property",
    "heartland_family": "Suburban Family",
    "working_class": "Working-Class",
}


def _assign_archetype(
    age: int, dwelling_type: str
) -> Literal["retiree", "student", "working_class", "heartland_family", "affluent_pro"]:
    """
    SimSG archetype from age, then home type (the same groups econometrics.py uses).
    Household income is not used: census incomes are for the whole household, so a young
    adult living with parents would otherwise look affluent.
      65+ -> retiree; 20-24 -> student (young adult); condo/landed -> affluent_pro;
      4 room, 5 room/exec -> heartland_family; 1-2 room, 3 room -> working_class
    """
    if age >= 65:
        return "retiree"
    if age <= 24:
        return "student"
    if dwelling_type in ("condo", "landed"):
        return "affluent_pro"
    if dwelling_type in ("4 room", "5 room/exec"):
        return "heartland_family"
    return "working_class"


def _draw_rows(rows: List[Dict[str, Any]], n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    weights = np.array([r["weight"] for r in rows], dtype=float)
    idx = rng.choice(len(rows), size=n, p=weights / weights.sum())
    return [rows[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Core Sampler Function
# ---------------------------------------------------------------------------
def sample_synthetic_cohort(
    planning_area: str,
    n: int = 30,
    seed: Optional[int] = None,
) -> List[SyntheticPersona]:
    """
    Draws a synthetic cohort of N adult residents of the planning area from the census tables.

    Parameters:
      planning_area: Target planning area (e.g. 'Ang Mo Kio'); case-insensitive
      n: Cohort sample size (default 30)
      seed: Optional integer seed; the same area, n and seed always give the same cohort

    Returns:
      List of SyntheticPersona Pydantic models.
    """
    households, people, eating_out_daily = _load_census_tables()

    areas = {a.lower(): a for a in list_supported_areas()}
    area_key = planning_area.lower().strip()
    if area_key not in areas:
        raise ValueError(
            f"Unsupported planning area '{planning_area}'. Supported areas: {', '.join(sorted(areas.values()))}"
        )
    formal_name = areas[area_key]
    area_code = _get_area_code(formal_name)

    # One independent random stream per step, so personas 1..k stay the same when n grows
    rng_household, rng_person, rng_detail = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )

    # 1. dwelling + income band (weighted by residents in the area)
    household_draws = _draw_rows(households[formal_name], n, rng_household)
    # 2. sex + age group, adults only
    person_draws = _draw_rows(people[formal_name], n, rng_person)

    cohort: List[SyntheticPersona] = []
    for i, (hh, person) in enumerate(zip(household_draws, person_draws)):
        # 3. exact age and income, uniform within the drawn group / band
        age = int(rng_detail.integers(person["age_lo"], person["age_hi"] + 1))
        monthly_income = round(float(rng_detail.uniform(hh["income_lo"], hh["income_hi"])), 2)
        dwelling_type = hh["dwelling"]

        cohort.append(
            SyntheticPersona(
                persona_id=f"{area_code}-{i + 1:03d}",
                planning_area=formal_name,
                age=age,
                age_bracket=person["age_group"],
                gender=person["sex"],
                dwelling_type=dwelling_type,
                monthly_household_income_sgd=monthly_income,
                income_bracket=hh["income_label"],
                archetype=_assign_archetype(age, dwelling_type),
                # 4. HES eating-out spend for the dwelling type
                daily_dining_budget_sgd=round(eating_out_daily[dwelling_type], 2),
            )
        )

    return cohort



def _synthesize_fallback_cohort_persona(
    cohort_name: str,
    concept: Optional[Any] = None,
) -> RepresentativePersona:
    """
    Targeted demographic synthesis when a cohort has 0 draws in a planning area
    (e.g. Working-Class in Bukit Timah, or Affluent in Outram).
    Guarantees that all 5 key Singapore demographics are ALWAYS portrayed.
    """
    cand_p = getattr(concept, "candidate_price_sgd", 4.50) if concept else 4.50

    if cohort_name == "Senior / Retiree":
        return RepresentativePersona(
            demographic_cohort="Senior / Retiree",
            persona_id="SYN-SNR-01",
            age=68,
            gender="Female",
            dwelling_type="3 room",
            monthly_income_sgd=2200.0,
            daily_dining_budget_sgd=14.50,
            buy_probability_pct=15 if cand_p > 3.0 else 65,
            sentiment="Negative" if cand_p > 3.0 else "Positive",
            price_ceiling_sgd=3.50,
            primary_objection="Fixed retirement allowance; significantly higher than coffeeshop kopi baseline.",
            quote=f"Aiya, S${cand_p:.2f} is very steep lah. Coffeeshop kopi only S$1.40, why spend so much?",
            too_cheap=0.80,
            cheap=1.40,
            expensive=2.80,
            too_expensive=3.50,
        )
    elif cohort_name == "Working-Class":
        return RepresentativePersona(
            demographic_cohort="Working-Class",
            persona_id="SYN-WRK-01",
            age=38,
            gender="Male",
            dwelling_type="3 room",
            monthly_income_sgd=3200.0,
            daily_dining_budget_sgd=18.00,
            buy_probability_pct=25 if cand_p > 4.5 else 60,
            sentiment="Negative" if cand_p > 4.5 else "Neutral",
            price_ceiling_sgd=4.50,
            primary_objection="Daily expenditure constraint; prefers hawker routine for weekday meals.",
            quote=f"Can buy once in a while as a treat, but S${cand_p:.2f} daily will burst my lunch budget.",
            too_cheap=1.20,
            cheap=2.20,
            expensive=3.80,
            too_expensive=4.50,
        )
    elif cohort_name == "Suburban Family":
        return RepresentativePersona(
            demographic_cohort="Suburban Family",
            persona_id="SYN-FAM-01",
            age=44,
            gender="Female",
            dwelling_type="4 room",
            monthly_income_sgd=8200.0,
            daily_dining_budget_sgd=28.00,
            buy_probability_pct=55 if cand_p <= 6.0 else 30,
            sentiment="Neutral" if cand_p <= 6.0 else "Negative",
            price_ceiling_sgd=6.00,
            primary_objection="Needs family bundle value and kid-friendly breakfast/tea sets.",
            quote=f"Would visit on weekends with the family if there are sharing sets or toast bundles.",
            too_cheap=1.80,
            cheap=3.50,
            expensive=5.20,
            too_expensive=6.00,
        )
    elif cohort_name == "Young Adult / Student":
        return RepresentativePersona(
            demographic_cohort="Young Adult / Student",
            persona_id="SYN-YNG-01",
            age=22,
            gender="Female",
            dwelling_type="4 room",
            monthly_income_sgd=1400.0,
            daily_dining_budget_sgd=14.00,
            buy_probability_pct=60 if cand_p <= 6.5 else 25,
            sentiment="Positive" if cand_p <= 6.5 else "Negative",
            price_ceiling_sgd=6.50,
            primary_objection="Attractive specialty aesthetic, but high for non-income student routine.",
            quote=f"Looks very aesthetic for study sessions, but S${cand_p:.2f} is slightly high for everyday.",
            too_cheap=1.50,
            cheap=3.20,
            expensive=5.50,
            too_expensive=6.50,
        )
    else:  # Affluent Private Property
        return RepresentativePersona(
            demographic_cohort="Affluent Private Property",
            persona_id="REP-AFF-01",
            age=36,
            gender="Male",
            dwelling_type="condo",
            monthly_income_sgd=15500.0,
            daily_dining_budget_sgd=55.00,
            buy_probability_pct=80 if cand_p <= 9.0 else 45,
            sentiment="Positive" if cand_p <= 9.0 else "Neutral",
            price_ceiling_sgd=9.00,
            primary_objection="Must maintain premium bean quality and extraction to justify pricing over chain cafes.",
            quote=f"Happy to pay S${cand_p:.2f} if the roast profile and single-origin quality meet specialty standards.",
            too_cheap=2.50,
            cheap=4.80,
            expensive=7.80,
            too_expensive=9.00,
        )


def select_representative_demographic_personas(
    personas: Sequence[SyntheticPersona],
    evaluations: Sequence[Any],
    concept: Optional[Any] = None,
) -> List[RepresentativePersona]:
    """
    Selects a selective JSON list of N=5 resident personas where each persona is
    from a different unique demographic such that all 5 key Singapore demographics are portrayed:
      1. Senior / Retiree
      2. Working-Class
      3. Suburban Family
      4. Young Adult / Student
      5. Affluent Private Property

    Formatted cleanly with flat, typed primitive fields ready for a React dashboard table.
    """
    TARGET_COHORTS = [
        "Senior / Retiree",
        "Working-Class",
        "Suburban Family",
        "Young Adult / Student",
        "Affluent Private Property",
    ]

    eval_map: Dict[str, Dict[str, Any]] = {}
    for ev in evaluations:
        ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else (ev if isinstance(ev, dict) else dict(ev))
        pid = ev_dict.get("persona_id")
        if pid:
            eval_map[pid] = ev_dict

    def score_fit(p: SyntheticPersona, cohort: str) -> float:
        age = p.age
        d = p.dwelling_type.lower()
        inc = p.monthly_household_income_sgd
        arch = getattr(p, "archetype", "").lower()

        if cohort == "Senior / Retiree":
            if age >= 65 or arch == "retiree":
                return 100.0 + age
            return -1.0
        elif cohort == "Young Adult / Student":
            if 20 <= age <= 24 or arch == "student":
                return 100.0 - age
            return -1.0
        elif cohort == "Affluent Private Property":
            if d in ["condo", "landed"]:
                return 100.0 + min(50.0, inc / 500.0)
            if arch == "affluent_pro":
                return 50.0 + min(50.0, inc / 500.0)
            return -1.0
        elif cohort == "Suburban Family":
            if d in ["4 room", "5 room/exec"] and 25 <= age < 65:
                return 100.0 + min(50.0, inc / 500.0)
            if arch == "heartland_family":
                return 70.0
            return -1.0
        elif cohort == "Working-Class":
            if d in ["1-2 room", "3 room"] and 25 <= age < 65:
                return 100.0 + max(0.0, (7000.0 - inc) / 100.0)
            if arch == "working_class":
                return 70.0
            return -1.0
        return -1.0

    selected: List[RepresentativePersona] = []
    used_ids: Set[str] = set()

    for cohort_name in TARGET_COHORTS:
        candidates = []
        for p in personas:
            if p.persona_id in used_ids:
                continue
            s = score_fit(p, cohort_name)
            if s > 0:
                candidates.append((s, p))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if candidates:
            best_s, best_p = candidates[0]
            used_ids.add(best_p.persona_id)
            ev_data = eval_map.get(best_p.persona_id, {})

            price_ceiling = (
                ev_data.get("expensive")
                or ev_data.get("too_expensive")
                or round(best_p.daily_dining_budget_sgd * 0.4, 2)
            )

            quote_val = str(
                ev_data.get("in_character_quote")
                or ev_data.get("quote")
                or "Considering the offering."
            )
            objection_val = str(
                ev_data.get("primary_objection")
                or "Price disparity vs hawker alternatives."
            )
            sentiment_val = str(
                ev_data.get("sentiment_label")
                or ev_data.get("sentiment")
                or "Neutral"
            )

            rep = RepresentativePersona(
                demographic_cohort=cohort_name,
                persona_id=best_p.persona_id,
                age=best_p.age,
                gender=best_p.gender,
                dwelling_type=best_p.dwelling_type,
                monthly_income_sgd=round(best_p.monthly_household_income_sgd, 2),
                daily_dining_budget_sgd=round(best_p.daily_dining_budget_sgd, 2),
                buy_probability_pct=int(ev_data.get("buy_probability_pct", 50)),
                sentiment=sentiment_val,
                price_ceiling_sgd=float(round(price_ceiling, 2)),
                primary_objection=objection_val,
                quote=quote_val,
                too_cheap=ev_data.get("too_cheap"),
                cheap=ev_data.get("cheap"),
                expensive=ev_data.get("expensive"),
                too_expensive=ev_data.get("too_expensive"),
            )
            selected.append(rep)
        else:
            rep = _synthesize_fallback_cohort_persona(cohort_name, concept)
            selected.append(rep)

    return selected


def format_all_demographic_personas(
    personas: Sequence[SyntheticPersona],
    evaluations: Sequence[Any],
    concept: Optional[Any] = None,
) -> List[RepresentativePersona]:
    """
    Formats and returns all N evaluated resident personas mapped to their corresponding
    Singapore demographic cohorts, preserving all individual evaluations, quotes,
    budget thresholds, and sentiments for the React demographic view.
    """
    eval_map: Dict[str, Dict[str, Any]] = {}
    for ev in evaluations:
        ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else (ev if isinstance(ev, dict) else dict(ev))
        pid = ev_dict.get("persona_id")
        if pid:
            eval_map[pid] = ev_dict

    def classify_cohort(p: SyntheticPersona) -> str:
        # The group follows the archetype (age, then home type); no income cutoffs
        arch = getattr(p, "archetype", "")
        if arch in ARCHETYPE_COHORT:
            return ARCHETYPE_COHORT[arch]
        return ARCHETYPE_COHORT[_assign_archetype(p.age, p.dwelling_type)]

    all_personas: List[RepresentativePersona] = []

    for p in personas:
        cohort_name = classify_cohort(p)
        ev_data = eval_map.get(p.persona_id, {})

        price_ceiling = (
            ev_data.get("expensive")
            or ev_data.get("too_expensive")
            or round(p.daily_dining_budget_sgd * 0.4, 2)
        )

        quote_val = str(
            ev_data.get("in_character_quote")
            or ev_data.get("quote")
            or "Considering the offering."
        )
        objection_val = str(
            ev_data.get("primary_objection")
            or "Price disparity vs hawker alternatives."
        )
        sentiment_val = str(
            ev_data.get("sentiment_label")
            or ev_data.get("sentiment")
            or "Neutral"
        )

        rep = RepresentativePersona(
            demographic_cohort=cohort_name,
            persona_id=p.persona_id,
            age=p.age,
            gender=p.gender,
            dwelling_type=p.dwelling_type,
            monthly_income_sgd=round(p.monthly_household_income_sgd, 2),
            daily_dining_budget_sgd=round(p.daily_dining_budget_sgd, 2),
            buy_probability_pct=int(ev_data.get("buy_probability_pct", 50)),
            sentiment=sentiment_val,
            price_ceiling_sgd=float(round(price_ceiling, 2)),
            primary_objection=objection_val,
            quote=quote_val,
            too_cheap=ev_data.get("too_cheap"),
            cheap=ev_data.get("cheap"),
            expensive=ev_data.get("expensive"),
            too_expensive=ev_data.get("too_expensive"),
        )
        all_personas.append(rep)

    return all_personas


select_representative_personas = select_representative_demographic_personas

# ---------------------------------------------------------------------------
# CLI Execution Block
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimSG Synthetic Cohort Sampler: Sample statistically representative personas from SingStat census data."
    )
    parser.add_argument(
        "planning_area",
        nargs="?",
        default="Ang Mo Kio",
        help="Singapore Planning Area (default: 'Ang Mo Kio')",
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=30,
        help="Number of personas to sample (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic sampling",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON array of personas",
    )
    args = parser.parse_args()

    cohort = sample_synthetic_cohort(
        planning_area=args.planning_area,
        n=args.n,
        seed=args.seed,
    )

    if args.json:
        payload = [p.model_dump() for p in cohort]
        print(json.dumps(payload, indent=2))
        return

    # Print summary table
    print(f"\n==========================================================================================")
    print(f"SimSG Synthetic Cohort: {args.planning_area} (N={len(cohort)}, Seed={args.seed})")
    print(f"==========================================================================================")
    header = (
        f"{'ID':<9} {'Age':<4} {'Gen':<6} {'Dwelling':<12} {'Monthly Inc':<13} "
        f"{'Archetype':<17} {'Eat-out/Day':<11} {'Age group':<9}"
    )
    print(header)
    print("-" * len(header))
    for p in cohort[:15]:  # Preview first 15 personas
        print(
            f"{p.persona_id:<9} {p.age:<4} {p.gender:<6} {p.dwelling_type:<12} "
            f"S${p.monthly_household_income_sgd:<11,.0f} {p.archetype:<17} "
            f"S${p.daily_dining_budget_sgd:<9.2f} {p.age_bracket:<9}"
        )
    if len(cohort) > 15:
        print(f"... and {len(cohort) - 15} more personas (use --json for full output).")

    print("\n--- Cohort Distribution Summary ---")
    archetypes = [p.archetype for p in cohort]
    dwellings = [p.dwelling_type for p in cohort]
    genders = [p.gender for p in cohort]

    for label, items in [("Archetypes", archetypes), ("Dwellings", dwellings), ("Genders", genders)]:
        counts = {x: items.count(x) for x in set(items)}
        dist_str = ", ".join(f"{k}: {v} ({v / len(items):.0%})" for k, v in sorted(counts.items()))
        print(f"  {label:<18}: {dist_str}")


if __name__ == "__main__":
    main()
