"""
backend/tests/test_agents.py
------------------------------
Unit & Integration Tests for SimSG Agent Modules:
  - Deliverable 3: `backend/app/agents/execution_engine.py`
  - Deliverable 5: `backend/app/agents/synthesis_agent.py`
"""

import asyncio
import json
import re
import pytest
from types import SimpleNamespace
from typing import List

try:
    from app.schemas import (
        PersonaEvaluationOutput,
        ClusteredObjection,
        MacroSynthesisReport,
    )
    from app.agents.execution_engine import (
        SYSTEM_PROMPT,
        evaluate_persona_rule_based,
        repair_price_monotonicity,
        run_persona_evaluations,
        build_price_limits_prompt,
        build_purchase_prompt,
        hide_prices,
        PRICE_HIDDEN,
        _evaluate_single_persona,
        format_rag_context,
    )
    from app.agents.synthesis_agent import (
        generate_macro_synthesis,
        generate_macro_synthesis_rule_based,
        build_synthesis_prompt,
    )
    from app.services.sampler import sample_synthetic_cohort, SyntheticPersona, average_daily_eating_out_sgd
    from app.services.query_decomposer import decompose_query_rule_based, DecomposedConcept
    from app.services.econometrics import run_econometric_analysis, EconometricAnalysisResult
except ImportError:
    from backend.app.schemas import (
        PersonaEvaluationOutput,
        ClusteredObjection,
        MacroSynthesisReport,
    )
    from backend.app.agents.execution_engine import (
        SYSTEM_PROMPT,
        evaluate_persona_rule_based,
        repair_price_monotonicity,
        run_persona_evaluations,
        build_price_limits_prompt,
        build_purchase_prompt,
        hide_prices,
        PRICE_HIDDEN,
        _evaluate_single_persona,
        format_rag_context,
    )
    from backend.app.agents.synthesis_agent import (
        generate_macro_synthesis,
        generate_macro_synthesis_rule_based,
        build_synthesis_prompt,
    )
    from backend.app.services.sampler import sample_synthetic_cohort, SyntheticPersona, average_daily_eating_out_sgd
    from backend.app.services.query_decomposer import decompose_query_rule_based, DecomposedConcept
    from backend.app.services.econometrics import run_econometric_analysis, EconometricAnalysisResult


# ===========================================================================
# 1. Monotonicity Repair Tests
# ===========================================================================

def test_repair_price_monotonicity_already_monotonic():
    """Verify that correctly ordered thresholds remain monotonic and unchanged."""
    tc, c, exp, te = repair_price_monotonicity(1.50, 3.00, 5.00, 8.00)
    assert tc == 1.50
    assert c == 3.00
    assert exp == 5.00
    assert te == 8.00
    assert tc <= c < exp <= te


def test_repair_price_monotonicity_inverted_inputs():
    """Verify that inverted prices (e.g. LLM halluncination/reversal) are sorted and repaired."""
    # Complete inversion: 8.00, 5.00, 3.00, 1.00
    tc, c, exp, te = repair_price_monotonicity(8.00, 5.00, 3.00, 1.00)
    assert tc <= c
    assert c < exp
    assert exp <= te
    assert tc == 1.00
    assert c == 3.00
    assert exp == 5.00
    assert te == 8.00


def test_repair_price_monotonicity_equal_values():
    """Verify that identical values are adjusted so that cheap < expensive strictly holds."""
    tc, c, exp, te = repair_price_monotonicity(4.00, 4.00, 4.00, 4.00, min_step=0.25)
    assert tc <= c
    assert c < exp
    assert exp <= te
    assert c == 4.00
    assert exp == 4.25
    assert te == 4.25


def test_repair_price_monotonicity_negative_or_zero():
    """Verify non-positive inputs are clamped to sensible positive thresholds."""
    tc, c, exp, te = repair_price_monotonicity(-2.00, 0.0, -1.0, 3.00)
    assert tc >= 0.10
    assert tc <= c < exp <= te


# ===========================================================================
# 2. PersonaEvaluationOutput Model Tests
# ===========================================================================

def test_persona_evaluation_output_schema_clamping():
    """Verify Pydantic field validators clamp buy probability and primary objection."""
    eval_out = PersonaEvaluationOutput(
        persona_id="AMK-001",
        buy_probability_pct=150,  # exceeds 100 -> should clamp to 100
        too_cheap=1.20,
        cheap=2.50,
        expensive=4.50,
        too_expensive=6.50,
        sentiment_label="positive",  # should capitalize
        in_character_quote="S$4.50 steady lah!",
        primary_objection="A" * 200,  # exceeds 120 chars -> should clamp to 120
    )
    assert eval_out.buy_probability_pct == 100
    assert eval_out.sentiment_label == "Positive"
    assert len(eval_out.primary_objection) == 120
    assert eval_out.too_cheap <= eval_out.cheap < eval_out.expensive <= eval_out.too_expensive


def test_persona_evaluation_output_negative_clamping():
    """Verify negative probability clamps to 0."""
    eval_out = PersonaEvaluationOutput(
        persona_id="AMK-002",
        buy_probability_pct=-25,
        too_cheap=1.00,
        cheap=2.00,
        expensive=3.50,
        too_expensive=5.00,
        sentiment_label="negative",
        in_character_quote="Wah lau so expensive!",
        primary_objection="Too costly",
    )
    assert eval_out.buy_probability_pct == 0
    assert eval_out.sentiment_label == "Negative"


# ===========================================================================
# 3. Prompt Construction & Formatting Tests
# ===========================================================================

def test_build_persona_prompt_contains_all_demographics():
    """Verify the generated prompt includes demographic attributes, budget, concept and RAG context."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=123)
    p = cohort[0]
    concept = decompose_query_rule_based("Artisanal Cold Brew Coffee for $6.50 in Ang Mo Kio")

    rag_chunks = [
        {"category": "pricing_benchmarks", "text": "Hawker kopi ranges from $1.20 to $1.50 in AMK Central."},
        {"category": "competitors_and_supply", "text": "Yakun and Toast Box operate at AMK Hub."},
    ]

    prompt = build_price_limits_prompt(p, concept, rag_chunks)

    # Demographic checks
    assert str(p.age) in prompt
    assert p.dwelling_type in prompt
    # The archetype is an internal label (it would imply a job or children); the AI gets the census facts only
    assert "Consumer Archetype" not in prompt
    assert p.archetype not in prompt
    assert f"S${p.daily_dining_budget_sgd:.2f}" in prompt
    # Fixed kopi/cafe reference prices were dropped (unsourced); local prices come from the RAG context
    # (checked by wording, not numbers: this pitch's own price S$6.50 equals the old cafe reference)
    assert "Reference Baselines" not in prompt
    assert "Traditional Hawker Kopi" not in prompt
    assert "Specialty Cafe Coffee" not in prompt
    # Speaking style was dropped (no research basis); the prompt must not carry it
    assert "Linguistic Style" not in prompt
    # Census budget is an HES average for the home type, labelled so the AI does not treat it as a cap
    assert p.age_bracket in prompt
    assert p.income_bracket in prompt
    assert "Typical Eating-Out Spend" in prompt
    assert "not your personal limit" in prompt
    assert "Daily Dining Budget" not in prompt

    # Concept checks: part 1 hides the seller's price everywhere, including inside the pitch text
    assert hide_prices(concept.concept_name) in prompt
    assert "Ang Mo Kio" in prompt
    assert "6.50" not in prompt
    assert "Candidate Price" not in prompt
    assert PRICE_HIDDEN in prompt
    assert "too_expensive" in prompt
    assert "buy_probability_pct" not in prompt

    # RAG checks: local prices stay, they are reference points rather than the seller's price
    assert "pricing_benchmarks" in prompt.lower() or "pricing norms" in prompt.lower()
    assert "Hawker kopi ranges from $1.20" in prompt


def test_purchase_prompt_reveals_price_and_asks_decision_only():
    p = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=123)[0]
    concept = decompose_query_rule_based("Artisanal Cold Brew Coffee for $6.50 in Ang Mo Kio")
    prompt = build_purchase_prompt(p, concept)
    assert "S$6.50" in prompt
    assert p.persona_id in prompt
    for field in ("buy_probability_pct", "sentiment_label", "in_character_quote", "primary_objection"):
        assert field in prompt
    for field in ("too_cheap", "`cheap`", "`expensive`", "too_expensive"):
        assert field not in prompt


@pytest.mark.parametrize(
    "pitch",
    [
        "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets",
        "Cold brew kiosk in Tampines for $6.50",
        "Tonkotsu ramen in Jurong East for SGD 12.50",
        "Bubble tea at 4.5 dollars in Outram",
    ],
)
def test_price_limits_prompt_never_shows_the_pitch_price(pitch):
    p = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=1)[0]
    concept = decompose_query_rule_based(pitch)
    prompt = build_price_limits_prompt(p, concept, rag_chunks=None)
    price = concept.candidate_price_sgd
    for shown in (f"{price:.2f}", f"{price:g}"):
        amount = re.escape(shown) + r"(?![\d])"
        assert not re.search(r"\$" + amount, prompt), shown
        assert not re.search(amount + r"\s*dollars", prompt), shown
        assert not re.search(r"SGD\s*" + amount, prompt), shown


class _TwoPartFakeClient:
    """Fake AsyncOpenAI: answers part 1 with price limits and part 2 with a decision, recording each call."""

    def __init__(self):
        self.calls = []

        async def create(**kwargs):
            self.calls.append(kwargs["messages"])
            if len(kwargs["messages"]) == 2:
                content = {"persona_id": "x", "too_cheap": 2.0, "cheap": 3.5, "expensive": 6.0, "too_expensive": 8.0}
            else:
                content = {"persona_id": "x", "buy_probability_pct": 35, "sentiment_label": "Neutral",
                           "in_character_quote": "Maybe once in a while.", "primary_objection": "Price"}
            msg = SimpleNamespace(content=json.dumps(content))
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_two_part_interview_asks_limits_before_revealing_price():
    persona = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=5)[0]
    concept = decompose_query_rule_based("S$7 artisanal kopi bar in Ang Mo Kio")
    fake = _TwoPartFakeClient()
    result = asyncio.run(_evaluate_single_persona(
        asyncio.Semaphore(1), fake, persona, concept, rag_chunks=None, use_mock=False, model="m"))

    assert len(fake.calls) == 2
    part1, part2 = fake.calls
    # Part 1: system + price-hidden prompt only
    assert [m["role"] for m in part1] == ["system", "user"]
    assert not re.search(r"\$7(?:\.00)?(?![\d.])", part1[1]["content"])
    # Part 2: same conversation, with the persona's own part 1 answer, then the price
    assert [m["role"] for m in part2] == ["system", "user", "assistant", "user"]
    assert json.loads(part2[2]["content"])["expensive"] == 6.0
    assert "S$7.00" in part2[3]["content"]

    # Result combines part 1 limits and part 2 decision
    assert (result.too_cheap, result.cheap, result.expensive, result.too_expensive) == (2.0, 3.5, 6.0, 8.0)
    assert result.buy_probability_pct == 35
    assert result.in_character_quote == "Maybe once in a while."


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_two_part_interview_falls_back_when_a_part_fails(fail_on_call):
    """json_schema and json_object both fail on that call -> the persona uses the rule-based answer."""
    persona = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=5)[0]
    concept = decompose_query_rule_based("S$7 artisanal kopi bar in Ang Mo Kio")

    class AlwaysFailFrom(_TwoPartFakeClient):
        def __init__(self):
            super().__init__()
            inner = self.chat.completions.create

            async def create(**kwargs):
                if len(self.calls) + 1 >= fail_on_call:
                    self.calls.append(kwargs["messages"])
                    raise RuntimeError("simulated AI failure")
                return await inner(**kwargs)

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    result = asyncio.run(_evaluate_single_persona(
        asyncio.Semaphore(1), AlwaysFailFrom(), persona, concept, rag_chunks=None, use_mock=False, model="m"))
    assert result == evaluate_persona_rule_based(persona, concept)


# Ground rules taken from respondent.py, which the persona AI must always be told
PROMPT_GROUND_RULES = [
    "most real people say no to most new products",
    "say you would not buy",
    "Do not invent other facts about yourself",
    "for the whole household, combined, not just you",
]

# Singlish markers that must not appear now that quotes are standard English
SINGLISH = re.compile(r"\b(lah|lor|leh|wah|shiok|siao|okay-lah|kopitiam|steady)\b", re.IGNORECASE)


@pytest.mark.parametrize("rule", PROMPT_GROUND_RULES)
def test_system_prompt_contains_ground_rule(rule):
    assert rule in SYSTEM_PROMPT


def test_system_prompt_has_no_speaking_style():
    assert "linguistic style" not in SYSTEM_PROMPT.lower()


def test_system_prompt_has_no_reference_price_anchors():
    assert "reference price" not in SYSTEM_PROMPT.lower()


def test_system_prompt_treats_eating_out_spend_as_guide_not_cap():
    assert "daily dining budget" not in SYSTEM_PROMPT.lower()
    assert "not a hard limit" in SYSTEM_PROMPT


def _persona_with(dwelling: str, budget: float, archetype: str = "heartland_family") -> SyntheticPersona:
    return SyntheticPersona(
        persona_id="AMK-001", planning_area="Ang Mo Kio", age=40, age_bracket="40-44", gender="Female",
        dwelling_type=dwelling, monthly_household_income_sgd=8000.0, income_bracket="$8,000 to $8,999",
        archetype=archetype, daily_dining_budget_sgd=budget,
    )


def test_backup_budget_factor_uses_hes_average():
    """
    Backup fair price = base x (home factor x 0.65 + budget factor x 0.35), budget factor = budget / ~S$10.99.
    4-room (home factor 1.00) heartland family (drinks base S$3.20):
      budget at the average -> fair price S$3.20, so 'cheap' = 0.75 x 3.20 = S$2.40.
    """
    concept =decompose_query_rule_based("Kopi stall in Ang Mo Kio", override_price=3.0)
    avg = average_daily_eating_out_sgd()

    at_average = evaluate_persona_rule_based(_persona_with("4 room", avg), concept)
    assert at_average.cheap == pytest.approx(2.40, abs=0.01)
    assert at_average.too_expensive == pytest.approx(3.20 * 1.80, abs=0.01)


def test_backup_budget_changes_fair_price_across_census_range():
    """With census budgets (S$7.77-15.10) the budget still moves the price, instead of all hitting the 0.6 floor."""
    concept = decompose_query_rule_based("Kopi stall in Ang Mo Kio", override_price=3.0)
    low = evaluate_persona_rule_based(_persona_with("4 room", 7.77), concept)
    high = evaluate_persona_rule_based(_persona_with("4 room", 15.10), concept)
    assert low.cheap < high.cheap
    # 7.77 / 10.99 = 0.707 -> 3.20 x (0.65 + 0.707 x 0.35) x 0.75 = S$2.15
    assert low.cheap == pytest.approx(2.15, abs=0.01)


@pytest.mark.parametrize("price", [1.0, 4.5, 7.0, 25.0])  # covers Positive, Neutral and Negative answers
def test_backup_quotes_are_standard_english(price):
    """The no-key backup writes standard English quotes and objections for every persona."""
    concept = decompose_query_rule_based("Artisanal kopi bar in Ang Mo Kio", override_price=price)
    personas = sample_synthetic_cohort("Ang Mo Kio", n=60, seed=1)

    for persona in personas:
        result = evaluate_persona_rule_based(persona, concept)
        assert not SINGLISH.search(result.in_character_quote), result.in_character_quote
        assert not SINGLISH.search(result.primary_objection), result.primary_objection


def test_simulate_output_has_no_singlish(client):
    """A full no-key /api/simulate run: quotes, objections and grouped quotes are standard English."""
    payload = {
        "raw_query": "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets",
        "override_town": "Ang Mo Kio",
        "sample_size": 30,
        "seed": 12345,
    }
    data = client.post("/api/simulate", json=payload).json()
    texts = [p["quote"] for p in data["personas"]] + [p["primary_objection"] for p in data["personas"]]
    for cluster in data["recommendations"]["clustered_objections"]:
        texts += cluster["representative_quotes"]
    for text in texts:
        assert not SINGLISH.search(text), text


# ===========================================================================
# 4. Async Execution Engine Batch Evaluation Tests
# ===========================================================================

def test_run_persona_evaluations_batch_mock():
    """Verify batch evaluation of 30 synthetic personas with bounded concurrency (mock/rule-based)."""
    async def _run():
        cohort = sample_synthetic_cohort("Tampines", n=30, seed=42)
        assert len(cohort) == 30

        concept = decompose_query_rule_based("Specialty Sourdough Bakery selling croissants for $4.80 in Tampines")

        evaluations = await run_persona_evaluations(
            personas=cohort,
            concept=concept,
            rag_chunks=None,
            use_mock=True,
        )

        assert len(evaluations) == 30
        for idx, e in enumerate(evaluations):
            assert isinstance(e, PersonaEvaluationOutput)
            assert e.persona_id == cohort[idx].persona_id
            assert 0 <= e.buy_probability_pct <= 100
            assert e.sentiment_label in ("Positive", "Neutral", "Negative")
            assert len(e.primary_objection) <= 120
            assert len(e.in_character_quote) > 0
            # Check strict Van Westendorp monotonicity
            assert e.too_cheap <= e.cheap < e.expensive <= e.too_expensive

    asyncio.run(_run())


def test_run_persona_evaluations_empty_cohort():
    """Verify graceful handling when an empty list of personas is provided."""
    async def _run():
        concept = decompose_query_rule_based("Boba tea stall in Outram for $3.50")
        res = await run_persona_evaluations([], concept, use_mock=True)
        assert res == []

    asyncio.run(_run())


# ===========================================================================
# 5. Macro Synthesis Report Model Tests
# ===========================================================================

def test_macro_synthesis_report_schema_normalization():
    """Verify MacroSynthesisReport normalization and validation."""
    report = MacroSynthesisReport(
        verdict_badge="high risk - pivot required",  # should normalize
        executive_summary="Executive summary sentence 1. Sentence 2. Sentence 3. Sentence 4.",
        clustered_objections=[
            ClusteredObjection(
                theme="Price Disparity vs Traditional Hawker Alternatives",
                frequency_pct=55.4,
                representative_quotes=["Wah lau so ex!", "Rather drink kopi-o downstairs."],
            )
        ],
        recommended_pivot="Drop price to S$3.50 and accept CDC vouchers.",
        simulated_post_pivot_uplift=68.5,
    )

    assert report.verdict_badge == "HIGH RISK - PIVOT REQUIRED"
    assert report.simulated_post_pivot_uplift == 68.5
    assert len(report.clustered_objections) == 1
    assert report.clustered_objections[0].frequency_pct == 55.4


# ===========================================================================
# 6. Macro Synthesis Strategic Pivot Agent Tests
# ===========================================================================

def test_generate_macro_synthesis_rule_based_fallback():
    """Verify deterministic rule-based synthesis generator produces structured strategic advice."""
    async def _run():
        cohort = sample_synthetic_cohort("Ang Mo Kio", n=30, seed=42)
        concept = decompose_query_rule_based("Artisanal Pour-Over Kopi Bar selling for $7.50 in Ang Mo Kio")

        # Run evaluations
        evals = await run_persona_evaluations(cohort, concept, use_mock=True)

        # Run econometrics
        eval_dicts = [{**p.model_dump(), **e.model_dump()} for p, e in zip(cohort, evals)]
        econometrics = run_econometric_analysis(eval_dicts)

        # Generate synthesis
        synthesis = await generate_macro_synthesis(concept, evals, econometrics, use_mock=True)

        assert isinstance(synthesis, MacroSynthesisReport)
        assert synthesis.verdict_badge in ("VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED")
        assert len(synthesis.executive_summary) > 50
        assert len(synthesis.clustered_objections) >= 2
        for cluster in synthesis.clustered_objections:
            assert isinstance(cluster, ClusteredObjection)
            assert cluster.frequency_pct >= 0.0
            assert len(cluster.representative_quotes) > 0
        assert "CDC" in synthesis.recommended_pivot or "voucher" in synthesis.recommended_pivot.lower() or "pivot" in synthesis.recommended_pivot.lower()
        assert 0.0 <= synthesis.simulated_post_pivot_uplift <= 100.0

    asyncio.run(_run())


# ===========================================================================
# 7. End-to-End Simulation Pipeline Integration Test
# ===========================================================================

def test_full_pipeline_end_to_end_30_personas():
    """
    Simulates the entire focus group pipeline from pitch decomposition to macro synthesis:
      1. Pitch decomposition -> DecomposedConcept
      2. Cohort sampling -> 30 SyntheticPersonas
      3. Async batch execution -> 30 PersonaEvaluationOutputs
      4. Quantitative econometrics -> EconometricAnalysisResult (PSM + Cohorts + Net Sentiment)
      5. Macro synthesis -> MacroSynthesisReport (Viability verdict, pivot, uplift)
    """
    async def _run():
        # 1. Pitch
        pitch = "Healthy artisanal grain bowls in Bukit Merah selling for $14.50"
        concept = decompose_query_rule_based(pitch)
        assert concept.target_planning_area == "Bukit Merah"
        assert concept.candidate_price_sgd == 14.50

        # 2. Sampler
        cohort = sample_synthetic_cohort(concept.target_planning_area, n=30, seed=99)
        assert len(cohort) == 30

        # 3. Execution Engine
        evaluations = await run_persona_evaluations(
            personas=cohort,
            concept=concept,
            rag_chunks=None,
            use_mock=True,
        )
        assert len(evaluations) == 30

        # 4. Econometrics
        combined_records = [{**p.model_dump(), **e.model_dump()} for p, e in zip(cohort, evaluations)]
        econometrics = run_econometric_analysis(combined_records)
        assert isinstance(econometrics, EconometricAnalysisResult)
        assert 0.0 <= econometrics.mean_buy_intent_pct <= 100.0
        assert -1.0 <= econometrics.net_sentiment_score <= 1.0
        assert econometrics.psm_intersections.opp > 0

        # 5. Macro Synthesis
        synthesis = await generate_macro_synthesis(
            concept=concept,
            evaluations=evaluations,
            econometrics=econometrics,
            use_mock=True,
        )
        assert isinstance(synthesis, MacroSynthesisReport)
        # Price $14.50 in Bukit Merah is high, expecting Moderate or High Risk
        assert synthesis.verdict_badge in ("MODERATE RISK", "HIGH RISK - PIVOT REQUIRED")
        assert synthesis.simulated_post_pivot_uplift > econometrics.mean_buy_intent_pct

    asyncio.run(_run())
