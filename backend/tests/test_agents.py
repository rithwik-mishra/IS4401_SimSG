"""
backend/tests/test_agents.py
------------------------------
Unit & Integration Tests for SimSG Agent Modules:
  - Deliverable 3: `backend/app/agents/execution_engine.py`
  - Deliverable 5: `backend/app/agents/synthesis_agent.py`
"""

import asyncio
import pytest
from typing import List

try:
    from app.schemas import (
        PersonaEvaluationOutput,
        ClusteredObjection,
        MacroSynthesisReport,
    )
    from app.agents.execution_engine import (
        repair_price_monotonicity,
        run_persona_evaluations,
        build_persona_prompt,
        format_rag_context,
    )
    from app.agents.synthesis_agent import (
        generate_macro_synthesis,
        generate_macro_synthesis_rule_based,
        build_synthesis_prompt,
    )
    from app.services.sampler import sample_synthetic_cohort, SyntheticPersona
    from app.services.query_decomposer import decompose_query_rule_based, DecomposedConcept
    from app.services.econometrics import run_econometric_analysis, EconometricAnalysisResult
except ImportError:
    from backend.app.schemas import (
        PersonaEvaluationOutput,
        ClusteredObjection,
        MacroSynthesisReport,
    )
    from backend.app.agents.execution_engine import (
        repair_price_monotonicity,
        run_persona_evaluations,
        build_persona_prompt,
        format_rag_context,
    )
    from backend.app.agents.synthesis_agent import (
        generate_macro_synthesis,
        generate_macro_synthesis_rule_based,
        build_synthesis_prompt,
    )
    from backend.app.services.sampler import sample_synthetic_cohort, SyntheticPersona
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
    """Verify the generated prompt includes demographic attributes, budget, and baselines."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=1, seed=123)
    p = cohort[0]
    concept = decompose_query_rule_based("Artisanal Cold Brew Coffee for $6.50 in Ang Mo Kio")

    rag_chunks = [
        {"category": "pricing_benchmarks", "text": "Hawker kopi ranges from $1.20 to $1.50 in AMK Central."},
        {"category": "competitors_and_supply", "text": "Yakun and Toast Box operate at AMK Hub."},
    ]

    prompt = build_persona_prompt(p, concept, rag_chunks)

    # Demographic checks
    assert str(p.age) in prompt
    assert p.dwelling_type in prompt
    assert p.archetype in prompt
    assert f"S${p.daily_dining_budget_sgd:.2f}" in prompt
    assert f"S${p.hawker_baseline_kopi_sgd:.2f}" in prompt
    assert f"S${p.cafe_baseline_coffee_sgd:.2f}" in prompt
    assert p.linguistic_style in prompt

    # Concept checks
    assert concept.concept_name in prompt
    assert "S$6.50" in prompt
    assert "Ang Mo Kio" in prompt

    # RAG checks
    assert "pricing_benchmarks" in prompt.lower() or "pricing norms" in prompt.lower()
    assert "Hawker kopi ranges from $1.20" in prompt


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
