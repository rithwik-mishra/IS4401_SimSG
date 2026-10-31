"""
backend/app/services/__init__.py
--------------------------------
SimSG application services.
"""

try:
    from app.services.query_decomposer import (
        DecomposedConcept,
        DecomposedQuery,
        QueryDecomposer,
        decompose_pitch,
        decompose_query,
    )
    from app.services.sampler import (
        SyntheticPersona,
        sample_synthetic_cohort,
        select_representative_demographic_personas,
        select_representative_personas,
    )
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from app.services.econometrics import (
        EconometricAnalysisResult,
        PSMCurvePoint,
        PSMIntersectionPoints,
        CohortMetrics,
        compute_econometrics,
    )
except ImportError:
    from backend.app.services.query_decomposer import (
        DecomposedConcept,
        DecomposedQuery,
        QueryDecomposer,
        decompose_pitch,
        decompose_query,
    )
    from backend.app.services.sampler import (
        SyntheticPersona,
        sample_synthetic_cohort,
        select_representative_demographic_personas,
        select_representative_personas,
    )
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from backend.app.services.econometrics import (
        EconometricAnalysisResult,
        PSMCurvePoint,
        PSMIntersectionPoints,
        CohortMetrics,
        compute_econometrics,
    )

__all__ = [
    "DecomposedConcept",
    "DecomposedQuery",
    "QueryDecomposer",
    "decompose_pitch",
    "decompose_query",
    "SyntheticPersona",
    "sample_synthetic_cohort",
    "select_representative_demographic_personas",
    "select_representative_personas",
    "SimulationContextService",
    "get_simulation_context_service",
    "EconometricAnalysisResult",
    "PSMCurvePoint",
    "PSMIntersectionPoints",
    "CohortMetrics",
    "compute_econometrics",
]
