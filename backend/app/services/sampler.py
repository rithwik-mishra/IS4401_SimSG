"""
backend/app/services/sampler.py
---------------------------------
SimSG Synthetic Cohort Sampler:
Generates statistically representative synthetic cohorts of Singapore residents
conditioned on official SingStat census data (Layer 1) and HES expenditure benchmarks.

Usage (CLI):
    python backend/app/services/sampler.py "Ang Mo Kio" 30
    python -m backend.app.services.sampler "Tampines" 10 --seed 42

Usage (Python API):
    from backend.app.services.sampler import sample_synthetic_cohort
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=30, seed=42)
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple



import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution for seed data
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent.parent  # backend/
_DATA_DIR = _BACKEND_DIR / "data"

LAYER1_PATH = _DATA_DIR / "layer1_seed.json"
HES_PATH = _DATA_DIR / "hes_expenditure_benchmarks.json"


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SyntheticPersona(BaseModel):
    """
    Full demographic and financial profile for an individual synthetic resident in SimSG.
    All attributes are mathematically grounded in SingStat census distributions
    and HES 2023 expenditure benchmarks.
    """

    persona_id: str = Field(..., description="Unique persona identifier (e.g. 'AMK-001')")
    planning_area: str = Field(..., description="Target Singapore planning area")
    age: int = Field(..., ge=20, description="Adult age (>= 20 years)")
    age_bracket: str = Field(..., description="Life-stage bracket: 20-24, 25-34, 35-49, 50-64, or 65+")
    gender: Literal["Male", "Female"] = Field(..., description="Gender (50/50 draw)")
    dwelling_type: Literal[
        "1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"
    ] = Field(..., description="HDB or private dwelling category")
    monthly_household_income_sgd: float = Field(
        ..., ge=0.0, description="Estimated monthly household income in SGD"
    )
    income_bracket: str = Field(..., description="Standard SingStat household income bracket")
    home_language: str = Field(..., description="Primary language spoken at home")
    linguistic_style: Literal[
        "Colloquial Singlish", "Mixed Code-Switch", "Standard Singapore English"
    ] = Field(..., description="Persona conversational linguistic register")
    archetype: Literal[
        "retiree", "student", "working_class", "heartland_family", "affluent_pro"
    ] = Field(..., description="SimSG consumer archetype")
    daily_dining_budget_sgd: float = Field(
        ..., ge=0.0, description="Daily food/dining out allowance in SGD (monthly_budget * 0.73 / 30)"
    )
    hawker_baseline_kopi_sgd: float = Field(
        default=1.30, description="Reference baseline price for traditional kopi (S$1.30)"
    )
    cafe_baseline_coffee_sgd: float = Field(
        default=6.50, description="Reference baseline price for specialty cafe coffee (S$6.50)"
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
# Data Caching & Benchmarks Loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_seed_data() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Loads and caches layer1_seed.json and hes_expenditure_benchmarks.json."""
    if not LAYER1_PATH.exists():
        raise FileNotFoundError(f"Layer 1 seed data not found at {LAYER1_PATH}")
    if not HES_PATH.exists():
        raise FileNotFoundError(f"HES benchmarks not found at {HES_PATH}")

    l1_raw = json.loads(LAYER1_PATH.read_text(encoding="utf-8"))
    hes_raw = json.loads(HES_PATH.read_text(encoding="utf-8"))

    # Index Layer 1 data by case-insensitive planning area name
    areas_by_name: Dict[str, Dict[str, Any]] = {}
    for entry in l1_raw:
        areas_by_name[entry["planning_area"].lower()] = entry

    return areas_by_name, hes_raw


# ---------------------------------------------------------------------------
# Sampling Configuration and Priors
# ---------------------------------------------------------------------------
AREA_CODE_MAP: Dict[str, str] = {
    "ang mo kio": "AMK",
    "bukit merah": "BKM",
    "bukit timah": "BTM",
    "jurong east": "JRE",
    "outram": "OUT",
    "tampines": "TAM",
}

DWELLING_KEYS: List[Tuple[str, Literal["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]]] = [
    ("hdb_1_2_room_pct", "1-2 room"),
    ("hdb_3_room_pct", "3 room"),
    ("hdb_4_room_pct", "4 room"),
    ("hdb_5_room_exec_pct", "5 room/exec"),
    ("condo_apt_pct", "condo"),
    ("landed_pct", "landed"),
]

AGE_BRACKET_CONFIG: List[Tuple[str, str, Tuple[int, int]]] = [
    ("student_pct", "20-24", (20, 24)),
    ("early_career_pct", "25-34", (25, 34)),
    ("family_career_pct", "35-49", (35, 49)),
    ("mature_working_pct", "50-64", (50, 64)),
    ("retiree_pct", "65+", (65, 82)),
]

LANGUAGE_KEYS: List[Tuple[str, str]] = [
    ("english_pct", "English"),
    ("mandarin_pct", "Mandarin"),
    ("chinese_dialect_pct", "Chinese Dialect"),
    ("malay_pct", "Malay"),
    ("indian_pct", "Tamil / Indian Languages"),
]

INCOME_BRACKET_INFO: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "unemployed_pct": ("Below $1,000 / No Income", (0.0, 0.0)),
    "below3k_pct": ("$1,000 - $2,999", (1000.0, 2999.0)),
    "income_3k_to_5999_pct": ("$3,000 - $5,999", (3000.0, 5999.0)),
    "income_6k_to_9999_pct": ("$6,000 - $9,999", (6000.0, 9999.0)),
    "income_10k_to_17499_pct": ("$10,000 - $17,499", (10000.0, 17499.0)),
    "income_17500_plus_pct": ("$17,500+", (17500.0, 28000.0)),
}

# Empirical national affinity matrix P(income_band | dwelling_type)
DWELLING_INCOME_AFFINITY: Dict[str, Dict[str, float]] = {
    "1-2 room": {
        "unemployed_pct": 0.35,
        "below3k_pct": 0.35,
        "income_3k_to_5999_pct": 0.20,
        "income_6k_to_9999_pct": 0.08,
        "income_10k_to_17499_pct": 0.02,
        "income_17500_plus_pct": 0.00,
    },
    "3 room": {
        "unemployed_pct": 0.15,
        "below3k_pct": 0.22,
        "income_3k_to_5999_pct": 0.30,
        "income_6k_to_9999_pct": 0.20,
        "income_10k_to_17499_pct": 0.10,
        "income_17500_plus_pct": 0.03,
    },
    "4 room": {
        "unemployed_pct": 0.05,
        "below3k_pct": 0.10,
        "income_3k_to_5999_pct": 0.25,
        "income_6k_to_9999_pct": 0.32,
        "income_10k_to_17499_pct": 0.20,
        "income_17500_plus_pct": 0.08,
    },
    "5 room/exec": {
        "unemployed_pct": 0.03,
        "below3k_pct": 0.06,
        "income_3k_to_5999_pct": 0.15,
        "income_6k_to_9999_pct": 0.32,
        "income_10k_to_17499_pct": 0.30,
        "income_17500_plus_pct": 0.14,
    },
    "condo": {
        "unemployed_pct": 0.02,
        "below3k_pct": 0.03,
        "income_3k_to_5999_pct": 0.07,
        "income_6k_to_9999_pct": 0.18,
        "income_10k_to_17499_pct": 0.35,
        "income_17500_plus_pct": 0.35,
    },
    "landed": {
        "unemployed_pct": 0.01,
        "below3k_pct": 0.02,
        "income_3k_to_5999_pct": 0.04,
        "income_6k_to_9999_pct": 0.10,
        "income_10k_to_17499_pct": 0.28,
        "income_17500_plus_pct": 0.55,
    },
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


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Safely normalizes an array of probabilities so they sum to 1.0."""
    total = probs.sum()
    if total <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


# ---------------------------------------------------------------------------
# Core Sampler Function
# ---------------------------------------------------------------------------
def sample_synthetic_cohort(
    planning_area: str,
    n: int = 30,
    seed: Optional[int] = None,
) -> List[SyntheticPersona]:
    """
    Draws a synthetic cohort of N Singapore adult personas representative of
    the specified planning area, based on Layer 1 census tables and HES 2023 benchmarks.

    Parameters:
      planning_area: Target planning area (e.g. 'Ang Mo Kio', 'Tampines', 'Bukit Timah', etc.)
      n: Cohort sample size (default 30)
      seed: Optional integer seed for 100% reproducible sampling

    Returns:
      List of SyntheticPersona Pydantic models.
    """
    areas_by_name, hes_data = _load_seed_data()
    area_key = planning_area.lower().strip()

    if area_key not in areas_by_name:
        supported = [entry["planning_area"] for entry in areas_by_name.values()]
        raise ValueError(
            f"Unsupported planning area '{planning_area}'. Supported areas: {', '.join(supported)}"
        )

    area_data = areas_by_name[area_key]
    formal_name = area_data["planning_area"]
    area_code = _get_area_code(formal_name)

    # Deterministic or independent RNG stream
    rng = np.random.default_rng(seed)

    # 1. Dwelling type distribution
    dwell_keys, dwell_names = zip(*DWELLING_KEYS)
    dwell_probs = np.array([float(area_data["dwelling_distribution"].get(k, 0.0)) for k in dwell_keys])
    dwell_probs = _normalize_probs(dwell_probs)

    # 2. Age distribution
    age_keys, age_brackets, age_ranges = zip(*AGE_BRACKET_CONFIG)
    age_probs = np.array([float(area_data["age_distribution"].get(k, 0.0)) for k in age_keys])
    age_probs = _normalize_probs(age_probs)

    # 3. Home language distribution
    lang_keys, lang_names = zip(*LANGUAGE_KEYS)
    lang_probs = np.array([float(area_data["home_language"].get(k, 0.0)) for k in lang_keys])
    lang_probs = _normalize_probs(lang_probs)

    # 4. Area marginal income distribution
    area_income_weights = {
        k: float(area_data["income_distribution"].get(k, 0.0))
        for k in INCOME_BRACKET_INFO.keys()
    }

    # 5. National expenditure constants
    national_bench = hes_data.get("national_benchmarks", {})
    eating_out_pct = float(national_bench.get("eating_out_share_of_food_pct", 0.73))
    hawker_kopi = float(national_bench.get("coffeeshop_kopi_sgd", 1.30))
    cafe_coffee = float(national_bench.get("third_wave_cafe_black_coffee_sgd", 6.50))
    archetypes_bench = hes_data.get("archetypes", {})

    cohort: List[SyntheticPersona] = []

    for i in range(n):
        persona_id = f"{area_code}-{i + 1:03d}"

        # 50/50 gender draw
        gender: Literal["Male", "Female"] = "Male" if rng.random() < 0.5 else "Female"

        # Draw age bracket and sample specific adult age >= 20
        age_idx = int(rng.choice(len(age_keys), p=age_probs))
        age_bracket = age_brackets[age_idx]
        min_age, max_age = age_ranges[age_idx]
        age = int(rng.integers(min_age, max_age + 1))

        # Draw dwelling type
        dwell_idx = int(rng.choice(len(dwell_names), p=dwell_probs))
        dwelling_type = dwell_names[dwell_idx]

        # Condition income on dwelling type and area marginals
        affinities = DWELLING_INCOME_AFFINITY[dwelling_type]
        income_keys = list(INCOME_BRACKET_INFO.keys())
        raw_income_weights = np.array(
            [affinities.get(k, 0.01) * area_income_weights.get(k, 0.01) for k in income_keys]
        )
        conditioned_income_probs = _normalize_probs(raw_income_weights)

        inc_idx = int(rng.choice(len(income_keys), p=conditioned_income_probs))
        chosen_inc_key = income_keys[inc_idx]
        bracket_label, (inc_min, inc_max) = INCOME_BRACKET_INFO[chosen_inc_key]

        if inc_min == 0.0 and inc_max == 0.0:
            monthly_income = 0.0
        else:
            monthly_income = round(float(rng.uniform(inc_min, inc_max)), 2)

        # Draw home language
        lang_idx = int(rng.choice(len(lang_names), p=lang_probs))
        home_language = lang_names[lang_idx]

        # Determine Archetype based on census and HES benchmark definitions
        archetype: Literal["retiree", "student", "working_class", "heartland_family", "affluent_pro"]
        if age >= 65:
            archetype = "retiree"
        elif age <= 24 and monthly_income < 6000:
            archetype = "student"
        elif dwelling_type in ["condo", "landed"] or monthly_income >= 10000:
            archetype = "affluent_pro"
        elif dwelling_type in ["4 room", "5 room/exec"] and monthly_income >= 6000:
            archetype = "heartland_family"
        else:
            archetype = "working_class"

        # Determine Linguistic Style based on archetype, age, and language
        linguistic_style: Literal["Colloquial Singlish", "Mixed Code-Switch", "Standard Singapore English"]
        if archetype == "retiree":
            if home_language in ["Chinese Dialect", "Mandarin", "Malay"]:
                p_style = [0.70, 0.25, 0.05]
            else:
                p_style = [0.45, 0.35, 0.20]
        elif archetype == "student":
            p_style = [0.25, 0.50, 0.25]
        elif archetype == "working_class":
            p_style = [0.50, 0.40, 0.10]
        elif archetype == "heartland_family":
            p_style = [0.25, 0.50, 0.25]
        elif archetype == "affluent_pro":
            if dwelling_type in ["condo", "landed"]:
                p_style = [0.05, 0.30, 0.65]
            else:
                p_style = [0.10, 0.35, 0.55]
        else:
            p_style = [0.34, 0.33, 0.33]

        styles: List[Literal["Colloquial Singlish", "Mixed Code-Switch", "Standard Singapore English"]] = [
            "Colloquial Singlish",
            "Mixed Code-Switch",
            "Standard Singapore English",
        ]
        linguistic_style = rng.choice(styles, p=p_style)

        # Deterministic Daily Dining Budget from HES expenditure benchmarks
        monthly_food_budget = float(archetypes_bench.get(archetype, {}).get("monthly_food_budget_sgd", 500.0))
        daily_dining_budget = round(monthly_food_budget * eating_out_pct / 30.0, 2)

        persona = SyntheticPersona(
            persona_id=persona_id,
            planning_area=formal_name,
            age=age,
            age_bracket=age_bracket,
            gender=gender,
            dwelling_type=dwelling_type,
            monthly_household_income_sgd=monthly_income,
            income_bracket=bracket_label,
            home_language=home_language,
            linguistic_style=linguistic_style,
            archetype=archetype,
            daily_dining_budget_sgd=daily_dining_budget,
            hawker_baseline_kopi_sgd=hawker_kopi,
            cafe_baseline_coffee_sgd=cafe_coffee,
        )
        cohort.append(persona)

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
            if arch == "affluent_pro" or inc >= 10000:
                return 50.0 + min(50.0, inc / 500.0)
            return -1.0
        elif cohort == "Suburban Family":
            if d in ["4 room", "5 room/exec"] and 25 <= age < 65:
                return 100.0 + min(50.0, inc / 500.0)
            if arch == "heartland_family":
                return 70.0
            return -1.0
        elif cohort == "Working-Class":
            if d in ["1-2 room", "3 room"] and 25 <= age < 65 and inc < 7000:
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
        f"{'Archetype':<17} {'Dining/Day':<11} {'Linguistic Style':<25}"
    )
    print(header)
    print("-" * len(header))
    for p in cohort[:15]:  # Preview first 15 personas
        print(
            f"{p.persona_id:<9} {p.age:<4} {p.gender:<6} {p.dwelling_type:<12} "
            f"S${p.monthly_household_income_sgd:<11,.0f} {p.archetype:<17} "
            f"S${p.daily_dining_budget_sgd:<9.2f} {p.linguistic_style:<25}"
        )
    if len(cohort) > 15:
        print(f"... and {len(cohort) - 15} more personas (use --json for full output).")

    print("\n--- Cohort Distribution Summary ---")
    archetypes = [p.archetype for p in cohort]
    dwellings = [p.dwelling_type for p in cohort]
    styles = [p.linguistic_style for p in cohort]

    for label, items in [("Archetypes", archetypes), ("Dwellings", dwellings), ("Linguistic Styles", styles)]:
        counts = {x: items.count(x) for x in set(items)}
        dist_str = ", ".join(f"{k}: {v} ({v / len(items):.0%})" for k, v in sorted(counts.items()))
        print(f"  {label:<18}: {dist_str}")


if __name__ == "__main__":
    main()
