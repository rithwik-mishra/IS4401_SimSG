"""
backend/app/services/econometrics.py
-----------------------------------
Econometrics & Statistical Analysis Service for SimSG.

Pure NumPy / SciPy implementation for quantitative focus group analytics:
1. Van Westendorp Price Sensitivity Meter (PSM):
   - Computes empirical cumulative curves (Too Cheap, Cheap, Expensive, Too Expensive)
   - Solves intersection points: Point of Marginal Cheapness (PMC),
     Optimal Price Point (OPP), Indifference Price Point (IPP),
     and Point of Marginal Expensiveness (PME) using SciPy root-finding / interpolation.
2. Demographic Cohort Segmentation (5 Singapore target segments):
   - Senior / Retiree (Age >= 65)
   - Working-Class (Dwelling in ["1-2 room", "3 room"])
   - Suburban Family (Dwelling in ["4 room", "5 room/exec", "4-room", "5-room"])
   - Young Adult / Student (Age 20-24)
   - Affluent Private Property (Dwelling in ["condo", "landed"])
3. Net Sentiment Score (-1.0 to +1.0) & Likelihood to Buy Intent (0 to 100%).

Zero LLM calls: Pure vectorised, deterministic mathematical computation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import optimize
from scipy.interpolate import interp1d

logger = logging.getLogger("simsg.econometrics")


# ---------------------------------------------------------------------------
# Pydantic v2 Data Models
# ---------------------------------------------------------------------------

class PSMIntersectionPoints(BaseModel):
    """
    Key price points derived from Van Westendorp Price Sensitivity Meter (SGD).
    """
    pmc: float = Field(..., description="Point of Marginal Cheapness: intersection of Too Cheap and Expensive")
    opp: float = Field(..., description="Optimal Price Point: intersection of Too Cheap and Too Expensive")
    ipp: float = Field(..., description="Indifference Price Point: intersection of Cheap and Expensive")
    pme: float = Field(..., description="Point of Marginal Expensiveness: intersection of Cheap and Too Expensive")


class PSMCurvePoint(BaseModel):
    """
    Discrete price point with cumulative percentage responses (0.0 to 100.0%).
    Ready for React Recharts line graphs.
    """
    price: float = Field(..., description="Price step in SGD")
    too_cheap: float = Field(..., description="Cumulative % finding price Too Cheap (downward sloping)")
    cheap: float = Field(..., description="Cumulative % finding price Cheap / Bargain (downward sloping)")
    expensive: float = Field(..., description="Cumulative % finding price Expensive (upward sloping)")
    too_expensive: float = Field(..., description="Cumulative % finding price Too Expensive (upward sloping)")


class CohortMetrics(BaseModel):
    """
    Statistical aggregates for a demographic cohort.
    """
    cohort_name: str = Field(..., description="Demographic cohort identifier")
    count: int = Field(..., description="Number of personas in sample belonging to this cohort")
    population_share_pct: float = Field(..., description="Share of this cohort in the sample (0-100%)")
    mean_buy_intent_pct: float = Field(..., description="Average buy intent / purchase probability (0-100%)")
    median_price_ceiling_sgd: float = Field(..., description="Median price ceiling in SGD")


class EconometricAnalysisResult(BaseModel):
    """
    Full quantitative econometric report for a focus group simulation.
    """
    mean_buy_intent_pct: float = Field(..., description="Mean purchase intent across all personas (0-100%)")
    net_sentiment_score: float = Field(..., description="Net sentiment score from -1.0 (all negative) to +1.0 (all positive)")
    sentiment_breakdown: dict[str, int] = Field(..., description="Counts of Positive, Neutral, Negative responses")
    psm_intersections: PSMIntersectionPoints = Field(..., description="Van Westendorp intersection prices")
    psm_curves: list[PSMCurvePoint] = Field(..., description="15-20 price curve steps for visualization")
    cohort_breakdown: list[CohortMetrics] = Field(..., description="Metrics across 5 key demographic cohorts")


# ---------------------------------------------------------------------------
# Cohort Definitions & Parsing Utilities
# ---------------------------------------------------------------------------

COHORT_SENIOR = "Senior / Retiree"
COHORT_WORKING_CLASS = "Working-Class"
COHORT_SUBURBAN_FAMILY = "Suburban Family"
COHORT_YOUNG_ADULT = "Young Adult / Student"
COHORT_AFFLUENT = "Affluent Private Property"

TARGET_COHORTS = [
    COHORT_SENIOR,
    COHORT_WORKING_CLASS,
    COHORT_SUBURBAN_FAMILY,
    COHORT_YOUNG_ADULT,
    COHORT_AFFLUENT,
]

WORKING_CLASS_DWELLINGS = {
    "1-2 room", "3 room", "1-2-room", "3-room", "1-2room", "3room",
    "hdb 1-2 room", "hdb 3 room", "hdb 1- or 2-room flat", "hdb 3-room flat"
}

SUBURBAN_FAMILY_DWELLINGS = {
    "4 room", "5 room/exec", "4-room", "5-room", "5-room/exec", "5 room", "exec",
    "hdb 4 room", "hdb 5 room", "hdb 4-room flat", "hdb 5-room or executive flat"
}

AFFLUENT_DWELLINGS = {
    "condo", "landed", "condominium", "private apartment", "landed property",
    "condominium or private apartment"
}


def _parse_age(age_val: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts min and max age from int, float, or string like '65-69', '90+', '20-24'.
    Returns (min_age, max_age).
    """
    if age_val is None:
        return None, None
    if isinstance(age_val, (int, float)):
        if np.isnan(age_val):
            return None, None
        v = float(age_val)
        return v, v
    s = str(age_val).strip().lower()
    if "+" in s:
        digits = re.findall(r"\d+", s)
        if digits:
            lo = float(digits[0])
            return lo, 120.0
    if "-" in s:
        parts = s.split("-")
        digits_lo = re.findall(r"\d+", parts[0])
        digits_hi = re.findall(r"\d+", parts[1])
        if digits_lo and digits_hi:
            return float(digits_lo[0]), float(digits_hi[0])
    digits = re.findall(r"\d+", s)
    if digits:
        v = float(digits[0])
        return v, v
    return None, None


def _is_senior(row: Dict[str, Any]) -> bool:
    """Age >= 65"""
    arch = str(row.get("archetype", "")).strip().lower()
    if arch in {"retiree", "senior"}:
        return True
    min_age, _ = _parse_age(row.get("age"))
    if min_age is not None and min_age >= 65:
        return True
    min_ag, _ = _parse_age(row.get("age_group"))
    if min_ag is not None and min_ag >= 65:
        return True
    age_lo = row.get("age_lo")
    if age_lo is not None and not pd.isna(age_lo) and float(age_lo) >= 65:
        return True
    return False


def _is_young_adult(row: Dict[str, Any]) -> bool:
    """Age between 20 and 24"""
    arch = str(row.get("archetype", "")).strip().lower()
    if arch in {"student", "young_adult"}:
        return True
    min_age, max_age = _parse_age(row.get("age"))
    if min_age is not None and 20 <= min_age <= 24:
        return True
    min_ag, max_ag = _parse_age(row.get("age_group"))
    if min_ag is not None and min_ag == 20:
        return True
    age_lo = row.get("age_lo")
    if age_lo is not None and not pd.isna(age_lo) and float(age_lo) == 20:
        return True
    return False


def _is_working_class(row: Dict[str, Any]) -> bool:
    """Dwelling in ['1-2 room', '3 room']"""
    arch = str(row.get("archetype", "")).strip().lower()
    if arch == "working_class":
        return True
    d = str(row.get("dwelling", "")).strip().lower()
    if d in WORKING_CLASS_DWELLINGS:
        return True
    dt = str(row.get("dwelling_text", "")).strip().lower()
    return any(w in dt for w in ["1- or 2-room", "3-room", "1-2 room", "3 room"])


def _is_suburban_family(row: Dict[str, Any]) -> bool:
    """Dwelling in ['4 room', '5 room/exec', '4-room', '5-room']"""
    arch = str(row.get("archetype", "")).strip().lower()
    if arch in {"heartland_family", "suburban_family"}:
        return True
    d = str(row.get("dwelling", "")).strip().lower()
    if d in SUBURBAN_FAMILY_DWELLINGS:
        return True
    dt = str(row.get("dwelling_text", "")).strip().lower()
    return any(w in dt for w in ["4-room", "5-room", "executive flat", "4 room", "5 room"])


def _is_affluent(row: Dict[str, Any]) -> bool:
    """Dwelling in ['condo', 'landed']"""
    arch = str(row.get("archetype", "")).strip().lower()
    if arch in {"affluent_pro", "affluent"}:
        return True
    d = str(row.get("dwelling", "")).strip().lower()
    if d in AFFLUENT_DWELLINGS or any(k in d for k in ["condo", "landed"]):
        return True
    dt = str(row.get("dwelling_text", "")).strip().lower()
    return any(k in dt for k in ["condo", "landed", "private apartment"])


def _extract_buy_intent_pct(record: Dict[str, Any]) -> float:
    """
    Extracts purchase likelihood / buy probability percentage (0.0 to 100.0).
    Resilient across various field representations.
    """
    # 1. Direct percentage fields
    for field in ["buy_probability_pct", "buy_intent_pct", "purchase_intent_pct"]:
        v = record.get(field)
        if v is not None and not pd.isna(v):
            try:
                return float(np.clip(float(v), 0.0, 100.0))
            except (ValueError, TypeError):
                pass

    # 2. Probability in [0.0, 1.0]
    for field in ["buy_probability", "buy_intent", "probability"]:
        v = record.get(field)
        if v is not None and not pd.isna(v):
            try:
                fv = float(v)
                return float(np.clip(fv * 100.0 if fv <= 1.0 else fv, 0.0, 100.0))
            except (ValueError, TypeError):
                pass

    # 3. 5-point Likert scale (1 = 0%, 2 = 25%, 3 = 50%, 4 = 75%, 5 = 100%)
    likert = record.get("likelihood_1_to_5")
    if likert is not None and not pd.isna(likert):
        try:
            val = float(likert)
            return float(np.clip((val - 1.0) / 4.0 * 100.0, 0.0, 100.0))
        except (ValueError, TypeError):
            pass

    # 4. Binary would_buy
    wb = record.get("would_buy")
    if wb is not None and not pd.isna(wb):
        if isinstance(wb, str):
            wb = wb.strip().lower() in {"true", "yes", "1"}
        return 100.0 if bool(wb) else 0.0

    return 0.0


def _extract_price_ceiling(record: Dict[str, Any], preferred_field: Optional[str] = None) -> Optional[float]:
    """
    Extracts price ceiling in SGD (from 'expensive', 'too_expensive', or 'max_price_sgd').
    """
    candidate_fields = []
    if preferred_field:
        candidate_fields.append(preferred_field)
    candidate_fields.extend(["expensive", "too_expensive", "max_price_sgd", "price_ceiling"])

    for field in candidate_fields:
        v = record.get(field)
        if v is not None and not pd.isna(v):
            try:
                fv = float(v)
                if fv > 0.0:
                    return fv
            except (ValueError, TypeError):
                continue
    return None


def _classify_sentiment(record: Dict[str, Any]) -> str:
    """
    Classifies a response as 'Positive', 'Neutral', or 'Negative'
    without requiring LLM text generation.
    """
    # 1. Check explicit sentiment field if present
    s = record.get("sentiment") or record.get("sentiment_label")
    if s is not None and not pd.isna(s):
        s_str = str(s).strip().lower()
        if "pos" in s_str:
            return "Positive"
        if "neg" in s_str:
            return "Negative"
        if "neu" in s_str:
            return "Neutral"

    # 2. Check 5-point likelihood scale
    likert = record.get("likelihood_1_to_5")
    if likert is not None and not pd.isna(likert):
        try:
            val = float(likert)
            if val >= 4.0:
                return "Positive"
            if val <= 2.0:
                return "Negative"
            return "Neutral"
        except (ValueError, TypeError):
            pass

    # 3. Check binary would_buy
    wb = record.get("would_buy")
    if wb is not None and not pd.isna(wb):
        if isinstance(wb, str):
            wb = wb.strip().lower() in {"true", "yes", "1"}
        return "Positive" if bool(wb) else "Negative"

    return "Neutral"


# ---------------------------------------------------------------------------
# Van Westendorp Price Sensitivity Meter (PSM) Engine
# ---------------------------------------------------------------------------

def _to_record_dicts(data: Union[pd.DataFrame, Sequence[Dict[str, Any]], Sequence[Any]]) -> List[Dict[str, Any]]:
    """Converts DataFrame, list of dicts, or list of Pydantic models into a list of dicts."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if not data:
        return []
    first = data[0]
    if isinstance(first, dict):
        return list(data)  # type: ignore
    if hasattr(first, "model_dump"):
        return [item.model_dump() for item in data]
    if hasattr(first, "__dict__"):
        return [item.__dict__ for item in data]
    return []


def _find_curve_intersection(
    prices: np.ndarray,
    down_curve: np.ndarray,
    up_curve: np.ndarray,
) -> float:
    """
    Finds the exact intersection price between a downward-sloping curve and
    an upward-sloping curve using linear interpolation and SciPy root-finding.

    Guarantees robust clamping to bounds when curves do not cross within range.
    """
    if len(prices) == 0:
        return 0.0
    if len(prices) == 1:
        return float(prices[0])

    diff = down_curve - up_curve

    # Case 1: Exact zero exists on the grid
    exact_zeros = prices[np.isclose(diff, 0.0, atol=1e-5)]
    if len(exact_zeros) > 0:
        return float(np.mean(exact_zeros))

    # Case 2: No sign change - curves don't cross in window
    if np.all(diff > 0):
        # Downward curve stays above upward curve across whole domain
        return float(prices[-1])
    if np.all(diff < 0):
        # Upward curve stays above downward curve across whole domain
        return float(prices[0])

    # Case 3: Identify crossing intervals and solve root via SciPy / linear interpolation
    roots: List[float] = []
    for i in range(len(prices) - 1):
        d1, d2 = diff[i], diff[i + 1]
        p1, p2 = prices[i], prices[i + 1]

        # Crossing between p1 and p2: d1 and d2 have opposite signs
        if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
            try:
                # Continuous linear interpolation function over segment
                f_seg = interp1d([p1, p2], [d1, d2], kind="linear", fill_value="extrapolate")
                root = optimize.brentq(f_seg, p1, p2)
                roots.append(float(root))
            except Exception:
                # Analytic fallback linear interpolation
                t = d1 / (d1 - d2)
                root = p1 + t * (p2 - p1)
                roots.append(float(root))
        elif np.isclose(d1, 0.0, atol=1e-5):
            roots.append(float(p1))

    if roots:
        # If multiple discrete crossings occur due to empirical noise, take the median
        return float(np.median(roots))

    # Fallback to midpoint
    return float(np.median(prices))


def compute_psm(
    records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    num_steps: int = 20,
    price_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[PSMIntersectionPoints, List[PSMCurvePoint]]:
    """
    Computes Van Westendorp Price Sensitivity Meter (PSM) intersection points
    and discrete cumulative response curves.

    Parameters:
        records: Collection of evaluated persona dicts or DataFrame containing
                 'too_cheap', 'cheap', 'expensive', 'too_expensive'.
                 If only 'max_price_sgd' is present, plausible anchors are derived.
        num_steps: Number of discrete price points for Recharts visualization (15-20).
        price_bounds: Optional (min_price, max_price) override.

    Returns:
        (PSMIntersectionPoints, List[PSMCurvePoint])
    """
    recs = _to_record_dicts(records)
    tc_list: List[float] = []
    c_list: List[float] = []
    e_list: List[float] = []
    te_list: List[float] = []

    for r in recs:
        tc = r.get("too_cheap")
        c = r.get("cheap")
        e = r.get("expensive")
        te = r.get("too_expensive")

        # If 4 PSM values not directly provided, derive anchors from max_price_sgd
        if any(v is None or pd.isna(v) for v in [tc, c, e, te]):
            mp = r.get("max_price_sgd")
            if mp is not None and not pd.isna(mp) and float(mp) > 0:
                base = float(mp)
                tc = max(0.50, round(base * 0.45, 2))
                c = max(tc + 0.20, round(base * 0.70, 2))
                e = max(c + 0.20, round(base, 2))
                te = max(e + 0.50, round(base * 1.30, 2))

        # Check validity and parse
        try:
            if None not in (tc, c, e, te) and not any(pd.isna([tc, c, e, te])):
                v_tc, v_c, v_e, v_te = float(tc), float(c), float(e), float(te)
                # Ensure rational ordering per respondent
                sorted_vals = sorted([v_tc, v_c, v_e, v_te])
                if sorted_vals[-1] > 0.0:
                    tc_list.append(sorted_vals[0])
                    c_list.append(sorted_vals[1])
                    e_list.append(sorted_vals[2])
                    te_list.append(sorted_vals[3])
        except (ValueError, TypeError):
            continue

    n_valid = len(tc_list)

    # Handle edge case: no valid price evaluations
    if n_valid == 0:
        default_min, default_max = (1.0, 15.0) if price_bounds is None else price_bounds
        steps = np.linspace(default_min, default_max, max(15, min(num_steps, 20)))
        curves = [
            PSMCurvePoint(
                price=round(float(p), 2),
                too_cheap=0.0,
                cheap=0.0,
                expensive=0.0,
                too_expensive=0.0,
            )
            for p in steps
        ]
        intersections = PSMIntersectionPoints(
            pmc=round(default_min, 2),
            opp=round((default_min + default_max) / 2.0, 2),
            ipp=round((default_min + default_max) / 2.0, 2),
            pme=round(default_max, 2),
        )
        return intersections, curves

    tc_arr = np.array(tc_list)
    c_arr = np.array(c_list)
    e_arr = np.array(e_list)
    te_arr = np.array(te_list)

    if price_bounds is not None:
        p_min, p_max = price_bounds
    else:
        p_min = float(np.min(tc_arr))
        p_max = float(np.max(te_arr))

    # Ensure valid price range
    if np.isclose(p_min, p_max) or p_min >= p_max:
        p_min = max(0.5, p_min - 2.0)
        p_max = p_max + 2.0

    # Generate 15-20 price steps
    n_points = int(np.clip(num_steps, 15, 20))
    raw_prices = np.linspace(p_min, p_max, n_points)
    # Round prices to 2 decimal places and ensure strictly monotonic
    prices = np.unique(np.round(raw_prices, 2))
    if len(prices) < 15:
        # Re-interpolate if rounding reduced uniqueness
        prices = np.round(np.linspace(p_min, p_max, n_points), 3)

    # Van Westendorp Cumulative Response Percentages:
    # Too Cheap: % where too_cheap >= p (downward sloping with price)
    # Cheap: % where cheap >= p (downward sloping with price)
    # Expensive: % where expensive <= p (upward sloping with price)
    # Too Expensive: % where too_expensive <= p (upward sloping with price)
    tc_curve = np.mean(tc_arr[:, None] >= prices[None, :], axis=0) * 100.0
    c_curve = np.mean(c_arr[:, None] >= prices[None, :], axis=0) * 100.0
    e_curve = np.mean(e_arr[:, None] <= prices[None, :], axis=0) * 100.0
    te_curve = np.mean(te_arr[:, None] <= prices[None, :], axis=0) * 100.0

    # Build discrete curve points for Recharts
    curve_points: List[PSMCurvePoint] = []
    for i in range(len(prices)):
        curve_points.append(
            PSMCurvePoint(
                price=round(float(prices[i]), 2),
                too_cheap=round(float(tc_curve[i]), 1),
                cheap=round(float(c_curve[i]), 1),
                expensive=round(float(e_curve[i]), 1),
                too_expensive=round(float(te_curve[i]), 1),
            )
        )

    # Calculate Van Westendorp Key Intersection Points:
    # 1. Point of Marginal Cheapness (PMC): Too Cheap x Expensive
    pmc = _find_curve_intersection(prices, tc_curve, e_curve)

    # 2. Optimal Price Point (OPP): Too Cheap x Too Expensive
    opp = _find_curve_intersection(prices, tc_curve, te_curve)

    # 3. Indifference Price Point (IPP): Cheap x Expensive
    ipp = _find_curve_intersection(prices, c_curve, e_curve)

    # 4. Point of Marginal Expensiveness (PME): Cheap x Too Expensive
    pme = _find_curve_intersection(prices, c_curve, te_curve)

    intersections = PSMIntersectionPoints(
        pmc=round(pmc, 2),
        opp=round(opp, 2),
        ipp=round(ipp, 2),
        pme=round(pme, 2),
    )

    return intersections, curve_points


# ---------------------------------------------------------------------------
# Demographic Cohort Breakdown Engine
# ---------------------------------------------------------------------------

def compute_cohort_breakdown(
    records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    price_ceiling_field: Optional[str] = None,
) -> List[CohortMetrics]:
    """
    Segments evaluated personas into 5 defined Singapore demographic cohorts
    and calculates statistical metrics: count, population share %,
    mean buy intent %, and median price ceiling.

    Cohorts:
      1. Senior / Retiree (Age >= 65)
      2. Working-Class (Dwelling in ["1-2 room", "3 room"])
      3. Suburban Family (Dwelling in ["4 room", "5 room/exec", "4-room", "5-room"])
      4. Young Adult / Student (Age between 20 and 24)
      5. Affluent Private Property (Dwelling in ["condo", "landed"])
    """
    recs = _to_record_dicts(records)
    total_count = len(recs)

    cohort_filters = [
        (COHORT_SENIOR, _is_senior),
        (COHORT_WORKING_CLASS, _is_working_class),
        (COHORT_SUBURBAN_FAMILY, _is_suburban_family),
        (COHORT_YOUNG_ADULT, _is_young_adult),
        (COHORT_AFFLUENT, _is_affluent),
    ]

    breakdown: List[CohortMetrics] = []

    for name, pred in cohort_filters:
        matched = [r for r in recs if pred(r)]
        count = len(matched)
        share_pct = round((count / total_count) * 100.0, 2) if total_count > 0 else 0.0

        if count > 0:
            buy_intents = [_extract_buy_intent_pct(r) for r in matched]
            mean_intent = round(float(np.mean(buy_intents)), 2)

            price_ceilings = [
                pc for r in matched
                if (pc := _extract_price_ceiling(r, price_ceiling_field)) is not None
            ]
            if price_ceilings:
                median_ceiling = round(float(np.median(price_ceilings)), 2)
            else:
                median_ceiling = 0.0
        else:
            mean_intent = 0.0
            median_ceiling = 0.0

        breakdown.append(
            CohortMetrics(
                cohort_name=name,
                count=count,
                population_share_pct=share_pct,
                mean_buy_intent_pct=mean_intent,
                median_price_ceiling_sgd=median_ceiling,
            )
        )

    return breakdown


# ---------------------------------------------------------------------------
# Sentiment & Intent Aggregations
# ---------------------------------------------------------------------------

def compute_sentiment(
    records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
) -> Tuple[float, Dict[str, int]]:
    """
    Calculates net sentiment score (-1.0 to +1.0) and breakdown counts.
    Net Sentiment = (Positive - Negative) / Total
    """
    recs = _to_record_dicts(records)
    if not recs:
        return 0.0, {"Positive": 0, "Neutral": 0, "Negative": 0}

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in recs:
        cat = _classify_sentiment(r)
        counts[cat] += 1

    total = len(recs)
    net_score = round(float((counts["Positive"] - counts["Negative"]) / total), 2)
    net_score = float(np.clip(net_score, -1.0, 1.0))

    return net_score, counts


def compute_mean_buy_intent(
    records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
) -> float:
    """Computes overall sample mean buy intent percentage (0.0 to 100.0)."""
    recs = _to_record_dicts(records)
    if not recs:
        return 0.0
    intents = [_extract_buy_intent_pct(r) for r in recs]
    return round(float(np.mean(intents)), 2)


# ---------------------------------------------------------------------------
# Econometrics Service Class
# ---------------------------------------------------------------------------

class EconometricsService:
    """
    High-level econometric analysis service combining Van Westendorp PSM,
    demographic cohort segmentation, and sentiment analytics.
    """

    @staticmethod
    def compute_psm(
        records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        num_steps: int = 20,
        price_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[PSMIntersectionPoints, List[PSMCurvePoint]]:
        return compute_psm(records, num_steps=num_steps, price_bounds=price_bounds)

    @staticmethod
    def compute_cohort_breakdown(
        records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        price_ceiling_field: Optional[str] = None,
    ) -> List[CohortMetrics]:
        return compute_cohort_breakdown(records, price_ceiling_field=price_ceiling_field)

    @staticmethod
    def compute_sentiment(
        records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    ) -> Tuple[float, Dict[str, int]]:
        return compute_sentiment(records)

    @classmethod
    def analyze(
        cls,
        records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        num_steps: int = 20,
        price_bounds: Optional[Tuple[float, float]] = None,
        price_ceiling_field: Optional[str] = None,
    ) -> EconometricAnalysisResult:
        """
        Runs comprehensive econometric pipeline on simulation results.
        """
        recs = _to_record_dicts(records)

        mean_intent = compute_mean_buy_intent(recs)
        net_sentiment, sentiment_dist = compute_sentiment(recs)
        intersections, curves = compute_psm(recs, num_steps=num_steps, price_bounds=price_bounds)
        cohorts = compute_cohort_breakdown(recs, price_ceiling_field=price_ceiling_field)

        return EconometricAnalysisResult(
            mean_buy_intent_pct=mean_intent,
            net_sentiment_score=net_sentiment,
            sentiment_breakdown=sentiment_dist,
            psm_intersections=intersections,
            psm_curves=curves,
            cohort_breakdown=cohorts,
        )


def analyze_simulation(
    records: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
    num_steps: int = 20,
    price_bounds: Optional[Tuple[float, float]] = None,
    price_ceiling_field: Optional[str] = None,
) -> EconometricAnalysisResult:
    """Convenience top-level wrapper around EconometricsService.analyze()."""
    return EconometricsService.analyze(
        records,
        num_steps=num_steps,
        price_bounds=price_bounds,
        price_ceiling_field=price_ceiling_field,
    )


# Alias for compatibility with external references
run_econometric_analysis = analyze_simulation


def compute_econometrics(
    evaluations: Sequence[Any],
    personas: Optional[Sequence[Any]] = None,
    num_steps: int = 20,
    price_bounds: Optional[Tuple[float, float]] = None,
) -> EconometricAnalysisResult:
    """
    Computes econometric analysis combining evaluations and optional personas.
    Merges persona demographic fields into evaluation records for cohort segmentation.
    """
    recs = []
    persona_map = {}
    if personas:
        for p in personas:
            p_dict = p.model_dump() if hasattr(p, "model_dump") else (p if isinstance(p, dict) else dict(p))
            pid = p_dict.get("persona_id")
            if pid:
                persona_map[pid] = p_dict

    for ev in evaluations:
        ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else (ev if isinstance(ev, dict) else dict(ev))
        merged = dict(ev_dict)
        pid = ev_dict.get("persona_id")
        if pid and pid in persona_map:
            for k, v in persona_map[pid].items():
                if k not in merged:
                    merged[k] = v
        recs.append(merged)

    return analyze_simulation(recs, num_steps=num_steps, price_bounds=price_bounds)


def get_econometrics_service() -> EconometricsService:
    """Singleton getter for EconometricsService."""
    return EconometricsService()


# ---------------------------------------------------------------------------
# Standalone Verification and Demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 80)
    print("SimSG Econometrics & Price Sensitivity Meter (Pure NumPy / SciPy)")
    print("=" * 80)

    # 1. Generate realistic synthetic respondents across demographic archetypes
    np.random.seed(42)
    n_sample = 50

    sample_dwellings = [
        "1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"
    ]
    dwelling_weights = [0.10, 0.35, 0.30, 0.15, 0.06, 0.04]

    sample_age_groups = [
        "20-24", "25-29", "35-39", "45-49", "55-59", "65-69", "75-79", "85-89"
    ]
    age_weights = [0.08, 0.14, 0.20, 0.20, 0.14, 0.12, 0.08, 0.04]

    synthetic_records: List[Dict[str, Any]] = []

    for i in range(n_sample):
        dwelling = np.random.choice(sample_dwellings, p=dwelling_weights)
        age_group = np.random.choice(sample_age_groups, p=age_weights)

        # Baseline willingness to pay higher for affluent dwellings
        if dwelling in {"condo", "landed"}:
            base_price = np.random.uniform(6.5, 9.5)
            likelihood = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
        elif dwelling in {"1-2 room", "3 room"}:
            base_price = np.random.uniform(2.5, 5.0)
            likelihood = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:
            base_price = np.random.uniform(4.0, 7.0)
            likelihood = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])

        # Van Westendorp 4 price points
        p_tc = max(0.80, round(base_price - np.random.uniform(1.2, 2.2), 2))
        p_c = max(p_tc + 0.30, round(base_price - np.random.uniform(0.4, 1.0), 2))
        p_e = max(p_c + 0.30, round(base_price + np.random.uniform(0.5, 1.5), 2))
        p_te = max(p_e + 0.50, round(base_price + np.random.uniform(1.8, 3.5), 2))

        synthetic_records.append({
            "persona_id": f"TEST-{i+1:03d}",
            "age_group": age_group,
            "dwelling": dwelling,
            "too_cheap": p_tc,
            "cheap": p_c,
            "expensive": p_e,
            "too_expensive": p_te,
            "likelihood_1_to_5": likelihood,
            "would_buy": likelihood >= 3,
        })

    # 2. Run analysis
    result = analyze_simulation(synthetic_records, num_steps=20)

    # 3. Print Results
    print(f"\n[Overall Sample Summary (N={n_sample})]")
    print(f"  Mean Buy Intent:     {result.mean_buy_intent_pct:.2f}%")
    print(f"  Net Sentiment Score: {result.net_sentiment_score:+.2f}")
    print(f"  Sentiment Counts:    {result.sentiment_breakdown}")

    print("\n[Van Westendorp PSM Key Intersection Points]")
    print(f"  Point of Marginal Cheapness (PMC):      S${result.psm_intersections.pmc:.2f} (Too Cheap x Expensive)")
    print(f"  Optimal Price Point (OPP):              S${result.psm_intersections.opp:.2f} (Too Cheap x Too Expensive)")
    print(f"  Indifference Price Point (IPP):         S${result.psm_intersections.ipp:.2f} (Cheap x Expensive)")
    print(f"  Point of Marginal Expensiveness (PME):   S${result.psm_intersections.pme:.2f} (Cheap x Too Expensive)")
    print(f"  Acceptable Price Range:                 [S${result.psm_intersections.pmc:.2f} - S${result.psm_intersections.pme:.2f}]")

    print(f"\n[PSM Cumulative Curves ({len(result.psm_curves)} price steps)]")
    print(f"  {'Price':>8} | {'Too Cheap %':>12} | {'Cheap %':>10} | {'Expensive %':>12} | {'Too Exp %':>10}")
    print("  " + "-" * 62)
    for pt in result.psm_curves:
        print(f"  S${pt.price:>6.2f} | {pt.too_cheap:>11.1f}% | {pt.cheap:>9.1f}% | {pt.expensive:>11.1f}% | {pt.too_expensive:>9.1f}%")

    print("\n[Demographic Cohort Breakdown (5 Target Segments)]")
    print(f"  {'Cohort Name':<28} | {'Count':>5} | {'Share %':>8} | {'Mean Intent %':>14} | {'Med Ceiling':>12}")
    print("  " + "-" * 78)
    for c in result.cohort_breakdown:
        print(f"  {c.cohort_name:<28} | {c.count:>5} | {c.population_share_pct:>7.1f}% | {c.mean_buy_intent_pct:>13.2f}% | S${c.median_price_ceiling_sgd:>10.2f}")

    # 4. Self-verifications
    assert 15 <= len(result.psm_curves) <= 20, f"Expected 15-20 price curves, got {len(result.psm_curves)}"
    assert -1.0 <= result.net_sentiment_score <= 1.0, "Net sentiment out of bounds"
    assert len(result.cohort_breakdown) == 5, f"Expected 5 cohorts, got {len(result.cohort_breakdown)}"
    assert result.psm_intersections.pmc <= result.psm_intersections.pme, "PMC must be <= PME in standard PSM"

    print("\n[OK] All mathematical invariants and econometric self-tests passed cleanly!\n")
