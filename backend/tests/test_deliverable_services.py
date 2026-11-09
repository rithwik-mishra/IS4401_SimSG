"""
backend/tests/test_deliverable_services.py
-------------------------------------------
Comprehensive tests for:
  1. backend/app/services/query_decomposer.py
  2. backend/app/services/sampler.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from backend.app.services.query_decomposer import (
    ALLOWED_PLANNING_AREAS,
    DecomposedConcept,
    QueryDecomposer,
    decompose_pitch,
    decompose_query,
    decompose_query_rule_based,
)
from backend.app.services import sampler as sampler_module
from backend.app.services.sampler import (
    CENSUS_DIR,
    SyntheticPersona,
    average_daily_eating_out_sgd,
    list_supported_areas,
    sample_synthetic_cohort,
)


# ===========================================================================
# 1. Query Decomposer Tests
# ===========================================================================

def test_query_decomposer_rule_based_extraction():
    """Verify deterministic rule-based extractor with various pitch styles."""
    pitch = "Artisanal Single-Origin Kopi Bar in Tampines selling pour-overs for $7.00"
    res = decompose_query_rule_based(pitch)

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Tampines"
    assert res.candidate_price_sgd == 7.00
    assert res.category_type == "beverage"
    assert "Kopi" in res.concept_name or "Artisanal" in res.concept_name
    assert res.concept_description == pitch


def test_query_decomposer_leading_sgd_price_pattern():
    """Verify extraction when S$7 appears at the very beginning of the query."""
    pitch = "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"
    res = decompose_query_rule_based(pitch)

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.candidate_price_sgd == 7.00
    assert res.category_type == "beverage"
    assert res.concept_name == "Artisanal Single-Origin Kopi Bar"


def test_query_decomposer_bakery_and_default_price():
    """Verify bakery classification and price defaulting to 4.50."""
    pitch = "French sourdough bakery with fresh croissants in Bukit Timah"
    res = decompose_query_rule_based(pitch)

    assert res.target_planning_area == "Bukit Timah"
    assert res.candidate_price_sgd == 4.50  # Default when price omitted
    assert res.category_type == "bakery"


def test_query_decomposer_dining_and_retail():
    """Verify dining and retail category inference."""
    dining_pitch = "Authentic tonkotsu ramen and gyoza restaurant in Jurong East for S$12.50"
    res_dining = decompose_query_rule_based(dining_pitch)
    assert res_dining.target_planning_area == "Jurong East"
    assert res_dining.category_type == "full_service_dining"
    assert res_dining.candidate_price_sgd == 12.50

    retail_pitch = "Lifestyle bookstore and craft stationery gift shop in Outram"
    res_retail = decompose_query_rule_based(retail_pitch)
    assert res_retail.target_planning_area == "Outram"
    assert res_retail.category_type == "retail"
    assert res_retail.candidate_price_sgd == 4.50


def test_query_decomposer_overrides():
    """Verify that override_price and override_town take precedence."""
    pitch = "Specialty tea shop in Tampines for $5.00"
    res = decompose_query_rule_based(
        pitch,
        override_price=9.50,
        override_town="Bukit Merah",
    )

    assert res.target_planning_area == "Bukit Merah"
    assert res.candidate_price_sgd == 9.50
    assert res.category_type == "beverage"


def test_query_decomposer_default_town_fallback():
    """Verify unmentioned or unknown towns default to Ang Mo Kio."""
    pitch = "A cozy bubble tea stall selling brown sugar milk tea for $4.20"
    res = decompose_query_rule_based(pitch)
    assert res.target_planning_area == "Ang Mo Kio"

    # Explicit invalid override should also safely fallback to Ang Mo Kio
    res_inv = decompose_query_rule_based(pitch, override_town="Atlantis")
    assert res_inv.target_planning_area == "Ang Mo Kio"


def test_query_decomposer_service_dispatch():
    """Verify QueryDecomposer service class and functional entry points."""
    decomposer = QueryDecomposer()
    res = decomposer.decompose("Croissant cafe in Ang Mo Kio for $4.80")

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.category_type in ["bakery", "beverage"]
    assert res.candidate_price_sgd == 4.80

    # Serialization test
    dump = res.model_dump()
    assert "concept_name" in dump
    assert "concept_description" in dump
    assert "target_planning_area" in dump
    assert "candidate_price_sgd" in dump
    assert "category_type" in dump


def test_query_decomposer_empty_input():
    """Verify empty/blank pitches don't crash and return safe baseline."""
    res = decompose_query("")
    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.candidate_price_sgd == 4.50
    assert res.category_type == "beverage"



# ===========================================================================
# 2. Synthetic Cohort Sampler Tests (census tables in data/sheets/processed)
# ===========================================================================

DWELLINGS = ["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]
PERSONA_FIELDS = {
    "persona_id", "planning_area", "age", "age_bracket", "gender", "dwelling_type",
    "monthly_household_income_sgd", "income_bracket", "archetype", "daily_dining_budget_sgd",
    "in_labour_force", "living_arrangement",
}
REMOVED_FIELDS = {"home_language", "linguistic_style", "hawker_baseline_kopi_sgd", "cafe_baseline_coffee_sgd"}


def _read_csv(name):
    with (CENSUS_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _shares(counter: Counter) -> dict:
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def _hes_daily_eating_out() -> dict:
    return {
        r["dwelling"]: float(r["spend_per_member"]) / 30
        for r in _read_csv("spend_per_member_dwelling.csv")
        if r["category"] == "Food and Beverage Serving Services"
    }


@pytest.mark.parametrize("area", ALLOWED_PLANNING_AREAS)
def test_sampler_all_target_areas(area: str):
    """Sampling runs cleanly across all 6 target planning areas, with valid census values."""
    cohort = sample_synthetic_cohort(planning_area=area, n=15, seed=123)
    assert len(cohort) == 15
    for p in cohort:
        assert isinstance(p, SyntheticPersona)
        assert p.planning_area == area
        assert p.age >= 20
        assert p.gender in ["Male", "Female"]
        assert p.dwelling_type in DWELLINGS
        assert p.monthly_household_income_sgd >= 0.0
        assert p.archetype in ["retiree", "student", "working_class", "heartland_family", "affluent_pro"]
        assert p.daily_dining_budget_sgd > 0.0


def test_sampler_covers_all_census_areas():
    """Every area with both household and resident tables can be sampled (29 areas)."""
    areas = list_supported_areas()
    assert len(areas) == 29
    for area in areas:
        assert len(sample_synthetic_cohort(area, n=3, seed=1)) == 3


def test_sampler_area_name_is_case_insensitive():
    a = sample_synthetic_cohort("ang mo kio", n=5, seed=3)
    b = sample_synthetic_cohort("Ang Mo Kio", n=5, seed=3)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]
    assert a[0].planning_area == "Ang Mo Kio"


def test_sampler_persona_id_format():
    """Verify persona_id follows area prefix and zero-padded integer format."""
    cohort_amk = sample_synthetic_cohort("Ang Mo Kio", n=3, seed=1)
    assert [p.persona_id for p in cohort_amk] == ["AMK-001", "AMK-002", "AMK-003"]

    cohort_tam = sample_synthetic_cohort("Tampines", n=2, seed=1)
    assert [p.persona_id for p in cohort_tam] == ["TAM-001", "TAM-002"]


def test_sampler_age_within_census_age_group():
    """Age is a whole year inside the census 5-year group; '90+' is treated as 90-94."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=400, seed=42)
    for p in cohort:
        assert re.fullmatch(r"\d+-\d+|90\+", p.age_bracket), p.age_bracket
        if p.age_bracket == "90+":
            assert 90 <= p.age <= 94
        else:
            lo, hi = map(int, p.age_bracket.split("-"))
            assert lo >= 20
            assert lo <= p.age <= hi


def test_sampler_income_within_census_band():
    """Income lies inside the census band; 'S$35,000 or more' is treated as 35,000-39,999."""
    cohort = sample_synthetic_cohort("Bukit Timah", n=400, seed=8)
    labels = set()
    for p in cohort:
        inc = p.monthly_household_income_sgd
        label = p.income_bracket
        labels.add(label)
        if label.endswith("or more"):
            assert label == "S$35,000 or more"
            assert 35000 <= inc <= 39999
        elif label.startswith("below"):
            assert label == "below S$1,000"
            assert 0 <= inc < 1000
        else:
            assert re.fullmatch(r"S\$[\d,]+ to S\$[\d,]+", label), label
            lo, hi = (int(x.replace("S$", "").replace(",", "")) for x in label.split(" to "))
            assert lo <= inc <= hi, (label, inc)
    assert "below S$1,000" in labels and "S$35,000 or more" in labels  # both ends were drawn


def test_app_personas_carry_income_band():
    """The app's persona rows carry the census band (shown instead of the drawn exact amount)."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=50, seed=3)
    rows = sampler_module.format_all_demographic_personas(cohort, [])
    assert [r.income_bracket for r in rows] == [p.income_bracket for p in cohort]


def test_sampler_deterministic_seed_reproducibility():
    """The same area, n and seed give identical cohorts; a different seed gives a different one."""
    cohort_a = sample_synthetic_cohort("Bukit Merah", n=20, seed=999)
    cohort_b = sample_synthetic_cohort("Bukit Merah", n=20, seed=999)
    cohort_c = sample_synthetic_cohort("Bukit Merah", n=20, seed=1000)

    assert [p.model_dump() for p in cohort_a] == [p.model_dump() for p in cohort_b]
    assert [p.model_dump() for p in cohort_a] != [p.model_dump() for p in cohort_c]


def test_sampler_first_personas_stable_when_n_grows():
    """Persona 1..k are the same whether 10 or 30 are drawn with the same seed."""
    small = sample_synthetic_cohort("Tampines", n=10, seed=5)
    large = sample_synthetic_cohort("Tampines", n=30, seed=5)
    assert [p.model_dump() for p in small] == [p.model_dump() for p in large[:10]]


def test_sampler_dining_budget_is_hes_eating_out_per_day():
    """daily_dining_budget_sgd = HES 2023 eating-out spend per member for the dwelling / 30, x 1.04 (2025 prices)."""
    expected = _hes_daily_eating_out()  # HES 2023 prices, straight from the table
    assert round(expected["1-2 room"], 2) == 7.77
    assert round(expected["condo"], 2) == 15.10
    cohort = sample_synthetic_cohort("Jurong East", n=60, seed=55)
    for p in cohort:
        assert p.daily_dining_budget_sgd == round(expected[p.dwelling_type] * sampler_module.HES_PRICE_FACTOR_TO_2025, 2)


def test_hes_price_factor_is_applied_once():
    """The 2023 -> 2025 price factor is 1.04 in total (about +3% in 2024, +1.2% in 2025), applied once."""
    assert sampler_module.HES_PRICE_FACTOR_TO_2025 == 1.04
    raw = _hes_daily_eating_out()
    by_dwelling = {p.dwelling_type: p.daily_dining_budget_sgd for p in sample_synthetic_cohort("Tampines", n=400, seed=9)}
    assert by_dwelling["1-2 room"] == 8.08 and by_dwelling["condo"] == 15.70  # 7.77 and 15.10 x 1.04
    for dwelling, budget in by_dwelling.items():
        assert budget / raw[dwelling] == pytest.approx(1.04, abs=0.001)  # not 1.04 x 1.04


def test_average_daily_eating_out_is_mean_of_six_dwellings():
    expected = _hes_daily_eating_out()
    assert average_daily_eating_out_sgd() == pytest.approx(sum(expected[d] for d in DWELLINGS) / 6 * 1.04)
    assert average_daily_eating_out_sgd() == pytest.approx(10.99 * 1.04, abs=0.01)  # about S$11.43


ARCHETYPE_BY_DWELLING = {
    "1-2 room": "working_class", "3 room": "working_class",
    "4 room": "heartland_family", "5 room/exec": "heartland_family",
    "condo": "affluent_pro", "landed": "affluent_pro",
}


def test_sampler_archetype_rules():
    """
    Archetype = 65+ and not in the labour force -> retiree; 20-24 -> student; otherwise home type
    (so a 65+ still in the labour force is grouped by home type). Household income is not used.
    """
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=300, seed=11)
    assert any(p.age >= 65 and p.in_labour_force for p in cohort)  # the new case is exercised
    for p in cohort:
        if p.age >= 65 and not p.in_labour_force:
            expected = "retiree"
        elif p.age <= 24:
            expected = "student"
        else:
            expected = ARCHETYPE_BY_DWELLING[p.dwelling_type]
        assert p.archetype == expected


def test_archetype_ignores_household_income():
    """A 22-year-old in a S$13,000 household is a young adult; a S$15,000 3-room household is working class."""
    assert sampler_module._assign_archetype(22, "4 room") == "student"
    assert sampler_module._assign_archetype(40, "3 room") == "working_class"
    assert sampler_module._assign_archetype(40, "4 room") == "heartland_family"
    assert sampler_module._assign_archetype(40, "condo") == "affluent_pro"
    assert sampler_module._assign_archetype(70, "landed", in_labour_force=False) == "retiree"


def test_working_senior_is_grouped_by_home_type():
    assert sampler_module._assign_archetype(67, "4 room", in_labour_force=True) == "heartland_family"
    assert sampler_module._assign_archetype(72, "3 room", in_labour_force=True) == "working_class"
    assert sampler_module._assign_archetype(66, "condo", in_labour_force=True) == "affluent_pro"
    assert sampler_module._assign_archetype(62, "4 room", in_labour_force=False) == "heartland_family"  # under 65
    assert sampler_module._assign_archetype(70, "landed") == "retiree"  # status unknown: old rule


def _labour_force_rates() -> dict:
    return {(r["age_group"], r["sex"]): float(r["rate"]) for r in _read_csv("labour_force_rate_age_sex.csv")}


@pytest.mark.parametrize("area", ["Ang Mo Kio", "Tampines", "Bukit Timah"])
def test_archetype_shares_follow_census(area):
    """
    Archetype shares follow the census: age and home type are drawn independently, and each 65+
    is a retiree with probability 1 - labour force rate (age group, sex); working seniors join the
    home-type groups.
    """
    cohort = sample_synthetic_cohort(area, n=20000, seed=77)
    people = [r for r in _read_csv("resident_area_age_sex.csv") if r["area"] == area and int(r["age_lo"]) >= 20]
    total = sum(float(r["residents"]) for r in people)
    rates = _labour_force_rates()
    p_retiree = sum(
        float(r["residents"]) * (1 - rates[("70+" if int(r["age_lo"]) >= 70 else r["age_group"], r["sex"])])
        for r in people if int(r["age_lo"]) >= 65
    ) / total
    p_young = sum(float(r["residents"]) for r in people if int(r["age_lo"]) == 20) / total
    p_middle = 1 - p_retiree - p_young  # 25-64, plus 65+ still in the labour force

    dwell = Counter()
    for r in _read_csv("fitted_area_dwelling_income.csv"):
        if r["area"] == area:
            dwell[ARCHETYPE_BY_DWELLING[r["dwelling"]]] += float(r["prob_in_area"])
    dwell = _shares(dwell)

    expected = {"retiree": p_retiree, "student": p_young}
    for arch in ("working_class", "heartland_family", "affluent_pro"):
        expected[arch] = p_middle * dwell.get(arch, 0.0)
    drawn = _shares(Counter(p.archetype for p in cohort))
    for arch, share in expected.items():
        assert abs(drawn.get(arch, 0.0) - share) < 0.015, f"{area} {arch}: {drawn.get(arch, 0):.3f} vs {share:.3f}"


def test_app_groups_follow_archetype():
    """Each persona's group in the app is fixed by its archetype (no income cutoffs)."""
    cohort = sample_synthetic_cohort("Tampines", n=200, seed=3)
    rows = sampler_module.format_all_demographic_personas(cohort, [])
    for p, row in zip(cohort, rows):
        assert row.demographic_cohort == sampler_module.ARCHETYPE_COHORT[p.archetype]


# ---------------------------------------------------------------------------
# Labour force status (60+) and living arrangement (65+), from the pipeline's senior tables
# ---------------------------------------------------------------------------

def test_senior_fields_only_for_their_ages():
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=500, seed=21)
    for p in cohort:
        assert (p.in_labour_force is not None) == (p.age >= 60)
        assert (p.living_arrangement is not None) == (p.age >= 65)


def test_app_personas_carry_senior_details():
    """The app's persona rows show labour force status (60+) and living arrangement wording (65+), else null."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=200, seed=13)
    rows = sampler_module.format_all_demographic_personas(cohort, [])
    for p, row in zip(cohort, rows):
        assert row.in_labour_force == p.in_labour_force
        assert row.living_arrangement == sampler_module.living_arrangement_text(p.living_arrangement)
        if p.age < 60:
            assert row.in_labour_force is None and row.living_arrangement is None
    assert any(r.living_arrangement == "lives alone" for r in rows)


def test_existing_draws_unchanged_by_senior_streams():
    """Steps 1-4 (age, sex, home type, income) are the same as before step 13 for the same seed."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=4, seed=12345)
    assert [(p.persona_id, p.age, p.gender, p.dwelling_type, p.monthly_household_income_sgd) for p in cohort] == [
        ("AMK-001", 71, "Male", "condo", 31055.63),
        ("AMK-002", 45, "Female", "4 room", 13086.38),
        ("AMK-003", 55, "Female", "3 room", 5097.17),
        ("AMK-004", 30, "Male", "3 room", 9012.42),
    ]


def test_labour_force_draws_match_rates():
    """On a large draw, the share in the labour force matches the 2025 rate for each age group and sex."""
    rates = _labour_force_rates()
    cohort = sample_synthetic_cohort("Tampines", n=40000, seed=31)
    groups = {}
    for p in cohort:
        if p.in_labour_force is None:
            continue
        key = ("70+" if p.age >= 70 else p.age_bracket, p.gender)
        groups.setdefault(key, []).append(p.in_labour_force)
    assert set(groups) == {(g, s) for g in ("60-64", "65-69", "70+") for s in ("Male", "Female")}
    for key, flags in groups.items():
        assert abs(sum(flags) / len(flags) - rates[key]) < 0.03, f"{key}: {sum(flags) / len(flags):.3f} vs {rates[key]:.3f}"


def test_living_arrangement_draws_match_shares():
    """On a large draw, living arrangements match the 2025 shares for each age group (65-69 ... 85+)."""
    expected = {}
    for r in _read_csv("senior_living_arrangement_age.csv"):
        expected.setdefault(r["age_group"], {})[r["arrangement"]] = float(r["share"])
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=40000, seed=41)
    groups = {}
    for p in cohort:
        if p.living_arrangement is not None:
            groups.setdefault("85+" if p.age >= 85 else p.age_bracket, Counter())[p.living_arrangement] += 1
    assert set(groups) == set(expected)
    for group, counts in groups.items():
        drawn = _shares(counts)
        for key, share in expected[group].items():
            assert abs(drawn.get(key, 0.0) - share) < 0.04, f"{group} {key}: {drawn.get(key, 0):.3f} vs {share:.3f}"


def test_senior_tables_match_pipeline_output():
    """pipeline/clean_data.py rebuilds the two senior tables exactly from the raw downloads."""
    import importlib.util
    import pandas as pd

    path = Path(sampler_module.__file__).resolve().parents[2] / "pipeline" / "clean_data.py"
    spec = importlib.util.spec_from_file_location("clean_data", path)
    clean_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean_data)

    lf = clean_data.clean_labour_force()
    living, published = clean_data.clean_living_arrangement()
    pd.testing.assert_frame_equal(lf, pd.read_csv(CENSUS_DIR / "labour_force_rate_age_sex.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(living, pd.read_csv(CENSUS_DIR / "senior_living_arrangement_age.csv"), check_dtype=False)
    assert clean_data.run_senior_checks(lf, living, published)
    # Rates the sampler uses (60+) stay within 0-100%; known >100% cells are prime-age only
    assert lf.query("age_lo >= 60")["rate"].between(0, 1).all()


@pytest.mark.parametrize("area", ["Ang Mo Kio", "Bukit Timah"])
def test_sampler_matches_census_shares(area):
    """
    A large draw reproduces the census shares of sex and age group (drawn first, never filtered), and
    of dwelling and income band within the household-rule tolerances (see test_town_mix_matches_census).
    """
    n = 20000
    cohort = sample_synthetic_cohort(area, n=n, seed=2024)
    tol = 0.015  # ~3 standard errors for a 20,000 draw (sex, age group)
    tol_household = {"dwelling": 0.02, "income band": 0.01}
    known_gap = {("Bukit Timah", "below S$1,000"): 0.025}  # see BUKIT_TIMAH_LOW_BAND_GAP

    fitted = [r for r in _read_csv("fitted_area_dwelling_income.csv") if r["area"] == area]
    dwell_expected, band_expected = Counter(), Counter()
    for r in fitted:
        label = sampler_module._income_label(float(r["income_lo"]), float(r["income_hi"]) if r["income_hi"] else None)
        dwell_expected[r["dwelling"]] += float(r["prob_in_area"])
        band_expected[label] += float(r["prob_in_area"])
    dwell_expected, band_expected = _shares(dwell_expected), _shares(band_expected)

    people = [r for r in _read_csv("resident_area_age_sex.csv") if r["area"] == area and int(r["age_lo"]) >= 20]
    sex_expected, age_expected = Counter(), Counter()
    for r in people:
        sex_expected[r["sex"]] += float(r["residents"])
        age_expected[r["age_group"]] += float(r["residents"])
    sex_expected, age_expected = _shares(sex_expected), _shares(age_expected)

    drawn = {
        "dwelling": _shares(Counter(p.dwelling_type for p in cohort)),
        "income band": _shares(Counter(p.income_bracket for p in cohort)),
        "sex": _shares(Counter(p.gender for p in cohort)),
        "age group": _shares(Counter(p.age_bracket for p in cohort)),
    }
    expected = {"dwelling": dwell_expected, "income band": band_expected, "sex": sex_expected, "age group": age_expected}
    for name, exp in expected.items():
        for key in set(exp) | set(drawn[name]):
            gap = abs(drawn[name].get(key, 0.0) - exp.get(key, 0.0))
            limit = known_gap.get((area, key), tol_household.get(name, tol))
            assert gap < limit, f"{area} {name} {key}: drawn {drawn[name].get(key, 0):.3f} vs census {exp.get(key, 0):.3f}"


# ---------------------------------------------------------------------------
# Step 14: household drawn after the person, with 3 rules for people aged 65+
# ---------------------------------------------------------------------------

RAG_TOWNS = ["Ang Mo Kio", "Tampines", "Bukit Timah", "Jurong East", "Bukit Merah", "Outram"]
SMALL_FLATS = ("1-2 room", "3 room")
BIG_HOMES = ("4 room", "5 room/exec", "condo", "landed")
# Bukit Timah: national rates predict ~2.7% of adults are retired and living alone (who must be
# placed below S$1,000 of market income), but the town's census puts only 1.1% below S$1,000. Even after moving
# every other low draw out, its lowest band runs ~1.6 points above the census.
BUKIT_TIMAH_LOW_BAND_GAP = 0.025


def _census_mix(area):
    dwell, band = Counter(), Counter()
    for r in _read_csv("fitted_area_dwelling_income.csv"):
        if r["area"] == area:
            label = sampler_module._income_label(float(r["income_lo"]), float(r["income_hi"]) if r["income_hi"] else None)
            dwell[r["dwelling"]] += float(r["prob_in_area"])
            band[label] += float(r["prob_in_area"])
    return _shares(dwell), _shares(band)


@pytest.fixture(scope="module")
def big_draws():
    """40,000 personas per town with ChromaDB passages (shared by the step 14 checks)."""
    return {town: sample_synthetic_cohort(town, n=40000, seed=2026) for town in RAG_TOWNS}


@pytest.mark.parametrize("area", RAG_TOWNS)
def test_town_mix_matches_census(area, big_draws):
    """
    Agreed check: income bands within 1 point of the census (rebalanced), home types within 2 points
    (household rules move them slightly). Exception: Bukit Timah's lowest band (BUKIT_TIMAH_LOW_BAND_GAP).
    """
    cohort = big_draws[area]
    census_dwell, census_band = _census_mix(area)
    drawn_dwell = _shares(Counter(p.dwelling_type for p in cohort))
    drawn_band = _shares(Counter(p.income_bracket for p in cohort))
    for key, share in census_dwell.items():
        assert abs(drawn_dwell.get(key, 0.0) - share) < 0.02, f"{area} {key}: {drawn_dwell.get(key, 0):.3f} vs {share:.3f}"
    for key, share in census_band.items():
        limit = BUKIT_TIMAH_LOW_BAND_GAP if (area, key) == ("Bukit Timah", "below S$1,000") else 0.01
        assert abs(drawn_band.get(key, 0.0) - share) < limit, f"{area} {key}: {drawn_band.get(key, 0):.3f} vs {share:.3f}"


@pytest.mark.parametrize("area", RAG_TOWNS)
def test_household_rules_apply(area, big_draws):
    """Retired + alone -> below S$1,000; alone -> smaller flats; with spouse and child -> bigger homes."""
    cohort = big_draws[area]
    groups = {}
    for p in cohort:
        groups.setdefault(sampler_module._household_group(p.age, p.in_labour_force, p.living_arrangement), []).append(p)

    retired_alone = groups["alone_retired"]
    assert retired_alone and all(p.income_bracket == "below S$1,000" for p in retired_alone)

    def share(ps, kinds):
        return sum(p.dwelling_type in kinds for p in ps) / len(ps)

    alone = retired_alone + groups.get("alone_working", [])
    assert share(alone, SMALL_FLATS) > share(cohort, SMALL_FLATS)
    with_family = groups["with_family"]
    assert share(with_family, BIG_HOMES) >= share(cohort, BIG_HOMES)


def test_town_without_lowest_band_uses_its_lowest_band(recwarn):
    """Tanglin has no 'below S$1,000' rows; retired-alone personas take its lowest band, with no maths warnings."""
    cohort = sample_synthetic_cohort("Tanglin", n=20000, seed=4)
    retired_alone = [p for p in cohort
                     if sampler_module._household_group(p.age, p.in_labour_force, p.living_arrangement) == "alone_retired"]
    lowest = min(p.monthly_household_income_sgd for p in cohort)
    assert retired_alone
    assert len({p.income_bracket for p in retired_alone}) == 1  # one band: the lowest present
    assert all(p.monthly_household_income_sgd < lowest + 1000 for p in retired_alone)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_household_group_rules():
    group = sampler_module._household_group
    assert group(83, False, "alone") == "alone_retired"
    assert group(83, False, "domestic_worker_only") == "alone_retired"
    assert group(72, True, "alone") == "alone_working"
    assert group(70, False, "spouse_and_child") == "with_family"
    assert group(70, True, "spouse_only") == "rest"
    assert group(62, False, None) == "rest"
    assert group(40, None, None) == "rest"


def test_unfiltered_personas_unchanged_by_household_rules():
    """
    Draw order changed (person first, household second), but a persona with no rule keeps the same
    home and income for the same seed, unless it drew 'below S$1,000' and was rebalanced.
    The seed 12345 Ang Mo Kio personas below are 'rest' and not in the lowest band.
    """
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=4, seed=12345)
    assert [(p.dwelling_type, p.monthly_household_income_sgd, p.income_bracket) for p in cohort] == [
        ("condo", 31055.63, "S$30,000 to S$34,999"),
        ("4 room", 13086.38, "S$13,000 to S$13,999"),
        ("3 room", 5097.17, "S$5,000 to S$5,999"),
        ("3 room", 9012.42, "S$9,000 to S$9,999"),
    ]


def test_sampler_serialization_and_fields():
    """model_dump has the census fields only; the dropped style/language/reference-price fields are gone."""
    cohort = sample_synthetic_cohort("Bukit Timah", n=5, seed=10)
    for p in cohort:
        dumped = p.model_dump()
        assert set(dumped) == PERSONA_FIELDS
        assert not REMOVED_FIELDS & set(dumped)
        assert len(json.dumps(dumped)) > 0


def test_sampler_does_not_read_layer1_or_hes_benchmark_files(monkeypatch):
    """The census sampler works without layer1_seed.json / hes_expenditure_benchmarks.json."""
    forbidden = ("layer1_seed.json", "hes_expenditure_benchmarks.json", "layer1_master.csv")
    real_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self.name not in forbidden, f"sampler read {self.name}"
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    sampler_module._load_census_tables.cache_clear()
    try:
        assert len(sample_synthetic_cohort("Ang Mo Kio", n=10, seed=1)) == 10
    finally:
        sampler_module._load_census_tables.cache_clear()


def test_layer1_and_hes_files_moved_to_legacy():
    """Step 11: the unused layer1 / HES files live in app/legacy/data, not in the live data folder."""
    backend_dir = Path(sampler_module.__file__).resolve().parents[2]
    legacy_data = backend_dir / "app" / "legacy" / "data"
    for name in ("layer1_seed.json", "hes_expenditure_benchmarks.json", "layer1_master.csv"):
        assert not (backend_dir / "data" / name).exists(), f"{name} is back in backend/data"
        assert (legacy_data / name).exists(), f"{name} missing from app/legacy/data"
    assert not (backend_dir / "scripts" / "extract_layer1_seed.py").exists()
    assert (backend_dir / "app" / "legacy" / "extract_layer1_seed.py").exists()


def test_live_code_does_not_import_legacy():
    """backend/app/legacy holds backups only; no live module imports from it."""
    app_dir = Path(sampler_module.__file__).resolve().parents[1]
    offenders = []
    for path in app_dir.rglob("*.py"):
        if "legacy" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+[\w.]*legacy", text, re.MULTILINE):
            offenders.append(str(path))
    assert offenders == []


def test_sampler_invalid_area_raises():
    """Verify that requesting an unsupported area raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported planning area"):
        sample_synthetic_cohort("NonExistentTown123", n=10)
