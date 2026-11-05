export interface SimulateRequest {
  raw_query: string;
  override_price?: number | null;
  override_town?: string | null;
  sample_size?: number;
  seed?: number | null;
  use_mock?: boolean;
}

export interface ClusteredObjection {
  theme: string;
  frequency_pct: number;
  representative_quotes: string[];
}

export interface MacroSynthesisReport {
  verdict_badge: 'VIABLE' | 'MODERATE RISK' | 'HIGH RISK - PIVOT REQUIRED' | string;
  executive_summary: string;
  clustered_objections: ClusteredObjection[];
  recommended_pivot: string;
  simulated_post_pivot_uplift: number;
}

export interface RepresentativePersona {
  demographic_cohort:
    | 'Senior / Retiree'
    | 'Working-Class'
    | 'Suburban Family'
    | 'Young Adult / Student'
    | 'Affluent Private Property'
    | string;
  persona_id: string;
  age: number;
  gender: 'Female' | 'Male' | string;
  dwelling_type: string;
  monthly_income_sgd: number;
  /** Census household income-from-work band, e.g. "S$7,000 to S$7,999" (shown instead of the exact amount) */
  income_bracket?: string | null;
  daily_dining_budget_sgd: number;
  buy_probability_pct: number;
  sentiment: 'Positive' | 'Neutral' | 'Negative' | string;
  price_ceiling_sgd: number;
  primary_objection: string;
  quote: string;
  too_cheap: number;
  cheap: number;
  expensive: number;
  too_expensive: number;
}

export interface PSMIntersections {
  pmc: number; // Point of Marginal Cheapness
  opp: number; // Optimal Price Point
  ipp: number; // Indifference Price Point
  pme: number; // Point of Marginal Expensiveness
}

export interface PSMCurvePoint {
  price: number;
  too_cheap: number;
  cheap: number;
  expensive: number;
  too_expensive: number;
}

export interface CohortBreakdown {
  cohort_name: string;
  count: number;
  population_share_pct: number;
  mean_buy_intent_pct: number;
  median_price_ceiling_sgd: number;
}

export interface EconometricAnalysisResult {
  mean_buy_intent_pct: number;
  net_sentiment_score: number;
  sentiment_breakdown: {
    Positive: number;
    Neutral: number;
    Negative: number;
  };
  psm_intersections: PSMIntersections;
  psm_curves: PSMCurvePoint[];
  cohort_breakdown: CohortBreakdown[];
}

/** Which pipeline steps the AI answered in a run, and which used the rule-based backup. */
export interface AIUsage {
  pitch_reader: 'ai' | 'backup';
  persona_ai: number;
  persona_backup: number;
  report_writer: 'ai' | 'backup';
}

export interface SimulateResponse {
  recommendations: MacroSynthesisReport;
  personas: RepresentativePersona[];
  statistics: EconometricAnalysisResult;
  // Seed the backend used to draw personas; absent on the built-in sample result.
  seed?: number | null;
  ai_usage?: AIUsage | null;
}

