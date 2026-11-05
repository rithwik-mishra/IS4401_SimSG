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
    """daily_dining_budget_sgd = HES 2023 eating-out spend per member for the dwelling / 30."""
    expected = _hes_daily_eating_out()
    assert round(expected["1-2 room"], 2) == 7.77
    assert round(expected["condo"], 2) == 15.10
    cohort = sample_synthetic_cohort("Jurong East", n=60, seed=55)
    for p in cohort:
        assert p.daily_dining_budget_sgd == round(expected[p.dwelling_type], 2)


def test_average_daily_eating_out_is_mean_of_six_dwellings():
    expected = _hes_daily_eating_out()
    assert average_daily_eating_out_sgd() == pytest.approx(sum(expected[d] for d in DWELLINGS) / 6)
    assert average_daily_eating_out_sgd() == pytest.approx(10.99, abs=0.01)


ARCHETYPE_BY_DWELLING = {
    "1-2 room": "working_class", "3 room": "working_class",
    "4 room": "heartland_family", "5 room/exec": "heartland_family",
    "condo": "affluent_pro", "landed": "affluent_pro",
}


def test_sampler_archetype_rules():
    """Archetype = age first (65+ retiree, 20-24 student), then home type; household income is not used."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=300, seed=11)
    for p in cohort:
        if p.age >= 65:
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
    assert sampler_module._assign_archetype(70, "landed") == "retiree"


@pytest.mark.parametrize("area", ["Ang Mo Kio", "Tampines", "Bukit Timah"])
def test_archetype_shares_follow_census(area):
    """Archetype shares = P(age group) x P(home type), since age and home type are drawn independently."""
    cohort = sample_synthetic_cohort(area, n=20000, seed=77)
    people = [r for r in _read_csv("resident_area_age_sex.csv") if r["area"] == area and int(r["age_lo"]) >= 20]
    total = sum(float(r["residents"]) for r in people)
    p_retiree = sum(float(r["residents"]) for r in people if int(r["age_lo"]) >= 65) / total
    p_young = sum(float(r["residents"]) for r in people if int(r["age_lo"]) == 20) / total
    p_middle = 1 - p_retiree - p_young

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


@pytest.mark.parametrize("area", ["Ang Mo Kio", "Bukit Timah"])
def test_sampler_matches_census_shares(area):
    """A large draw reproduces the census shares of dwelling, income band, sex and age group."""
    n = 20000
    cohort = sample_synthetic_cohort(area, n=n, seed=2024)
    tol = 0.015  # ~3 standard errors for a 20,000 draw

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
            assert gap < tol, f"{area} {name} {key}: drawn {drawn[name].get(key, 0):.3f} vs census {exp.get(key, 0):.3f}"


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
