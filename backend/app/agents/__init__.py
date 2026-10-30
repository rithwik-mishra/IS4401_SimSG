# backend/app/agents/__init__.py
"""
SimSG Agents Module:
- Execution Engine: async concurrent persona focus group evaluator (Deliverable 3).
- Macro Synthesis Agent: strategic viability verdict & pivot generator (Deliverable 5).
"""

try:
    from app.agents.execution_engine import (
        PersonaEvaluationOutput,
        repair_price_monotonicity,
        run_persona_evaluations,
    )
    from app.agents.synthesis_agent import (
        ClusteredObjection,
        MacroSynthesisReport,
        generate_macro_synthesis,
    )
except ImportError:
    from backend.app.agents.execution_engine import (
        PersonaEvaluationOutput,
        repair_price_monotonicity,
        run_persona_evaluations,
    )
    from backend.app.agents.synthesis_agent import (
        ClusteredObjection,
        MacroSynthesisReport,
        generate_macro_synthesis,
    )

__all__ = [
    "PersonaEvaluationOutput",
    "repair_price_monotonicity",
    "run_persona_evaluations",
    "ClusteredObjection",
    "MacroSynthesisReport",
    "generate_macro_synthesis",
]
