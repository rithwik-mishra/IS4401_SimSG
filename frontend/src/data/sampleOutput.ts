import { SimulateResponse, RepresentativePersona } from '../types/api';

const generateSample30Personas = (): RepresentativePersona[] => {
  const cohorts = [
    { cohort: 'Senior / Retiree', dwelling: '3 room', income: 2100, budget: 11.5, ageBase: 68 },
    { cohort: 'Working-Class', dwelling: '3 room', income: 3400, budget: 16.0, ageBase: 36 },
    { cohort: 'Suburban Family', dwelling: '4 room', income: 8400, budget: 28.0, ageBase: 44 },
    { cohort: 'Young Adult / Student', dwelling: '4 room', income: 1200, budget: 14.0, ageBase: 22 },
    { cohort: 'Affluent Private Property', dwelling: 'condo', income: 16800, budget: 52.0, ageBase: 35 },
  ];

  const sampleQuotes = [
    "Artisanal kopi sounds appealing, but I need to see if the taste is worth the price.",
    "It is a bit steep for an everyday drink; I might consider it as a weekly treat with colleagues.",
    "If it accepted CDC vouchers, it would be easier to visit regularly.",
    "Specialty cold brew is nice, but hawker kopi is still my usual morning drink.",
    "If the beans are genuinely single-origin specialty grade, I am willing to pay.",
    "A toast set bundle at a reasonable price would attract our family on weekends.",
    "It looks like a pleasant place to study or meet friends, but I hope the portions are substantial.",
    "I would compare it with my neighbourhood coffeeshop first, where kopi is only S$1.50.",
  ];

  const objections = [
    "High price for daily morning routine compared to hawker alternatives.",
    "Needs value toast bundling or morning set discounts.",
    "Requires authentic quality justification over standard chain cafes.",
    "Fixed retirement allowance limits discretionary specialty spending.",
    "Student budget constraint; reserved for special social study outings.",
    "Must maintain consistent single-origin extraction standards.",
  ];

  const personas: RepresentativePersona[] = [];

  for (let i = 1; i <= 30; i++) {
    const c = cohorts[(i - 1) % cohorts.length];
    const id = `AMK-${String(i).padStart(3, '0')}`;
    const age = c.ageBase + ((i * 3) % 12) - 4;
    const gender = i % 2 === 0 ? 'Female' : 'Male';
    const isAffluent = c.cohort === 'Affluent Private Property';
    const isSenior = c.cohort === 'Senior / Retiree';

    const buyProb = isAffluent ? 85 + (i % 15) : isSenior ? 30 + (i % 25) : 55 + (i % 35);
    const sentiment = buyProb >= 70 ? 'Positive' : buyProb >= 45 ? 'Neutral' : 'Negative';

    personas.push({
      persona_id: id,
      demographic_cohort: c.cohort,
      age: Math.max(20, Math.min(84, age)),
      gender,
      dwelling_type: c.dwelling,
      monthly_income_sgd: Math.round(c.income * (0.85 + (i % 7) * 0.05)),
      daily_dining_budget_sgd: Math.round(c.budget * (0.9 + (i % 5) * 0.06) * 100) / 100,
      buy_probability_pct: Math.min(100, buyProb),
      sentiment,
      price_ceiling_sgd: isAffluent ? 8.5 : isSenior ? 4.5 : 6.5,
      primary_objection: objections[i % objections.length],
      quote: sampleQuotes[i % sampleQuotes.length],
      too_cheap: isAffluent ? 2.5 : 1.0,
      cheap: isAffluent ? 4.5 : 3.0,
      expensive: isAffluent ? 8.0 : 6.0,
      too_expensive: isAffluent ? 10.0 : 8.0,
    });
  }

  return personas;
};

export const SAMPLE_SIMULATION_RESULT: SimulateResponse = {
  recommendations: {
    verdict_badge: 'MODERATE RISK',
    executive_summary:
      'The Artisanal Single-Origin Kopi Bar is positioned at a candidate price of S$7.00, which is substantially above the optimal price point (OPP) of S$4.50. Despite solid interest from affluent and younger demographics, qualitative feedback reveals significant resistance among working-class residents and retirees in Ang Mo Kio. A strategic pivot focusing on price repositioning, morning bundles, and CDC voucher integration will dramatically improve product-market fit.',
    clustered_objections: [
      {
        theme: 'Price Sensitivity',
        frequency_pct: 60,
        representative_quotes: [
          'If CDC vouchers were accepted, I would be more likely to visit often.',
          'The price is too high for daily kopi; I need to save for other expenses.',
          'I am not sure I can afford it. Daily coffee at $7 is too steep.',
        ],
      },
      {
        theme: 'Quality vs Value Benchmark',
        frequency_pct: 30,
        representative_quotes: [
          'Will the taste justify the price compared to regular hawker kopi?',
          'Just hope the quality matches the premium price point.',
          'I am not sure the specialty cold brew can beat traditional sock-brewed kopi.',
        ],
      },
      {
        theme: 'Habit & Routine',
        frequency_pct: 10,
        representative_quotes: [
          'I prefer my usual coffeeshop, where the stall owner already knows my order.',
          'Artisanal kopi sounds appealing, but only as a weekend treat.',
        ],
      },
    ],
    recommended_pivot:
      'Reposition the standard cup to S$4.50 to align with the optimal price point (OPP), introduce a S$5.80 kaya toast set bundle, and register for CDC vouchers. Maintain the S$7.00 single-origin cold brew as a reserve premium item.',
    simulated_post_pivot_uplift: 78.5,
  },
  personas: generateSample30Personas(),
  statistics: {
    mean_buy_intent_pct: 91.67,
    net_sentiment_score: 0.67,
    sentiment_breakdown: {
      Positive: 22,
      Neutral: 6,
      Negative: 2,
    },
    psm_intersections: {
      pmc: 3.0,
      opp: 4.5,
      ipp: 4.25,
      pme: 5.75,
    },
    psm_curves: [
      { price: 0.5, too_cheap: 100, cheap: 100, expensive: 0, too_expensive: 0 },
      { price: 1.0, too_cheap: 70, cheap: 100, expensive: 0, too_expensive: 0 },
      { price: 1.5, too_cheap: 0, cheap: 100, expensive: 0, too_expensive: 0 },
      { price: 2.0, too_cheap: 0, cheap: 100, expensive: 0, too_expensive: 0 },
      { price: 2.5, too_cheap: 0, cheap: 93.3, expensive: 0, too_expensive: 0 },
      { price: 3.0, too_cheap: 0, cheap: 86.7, expensive: 0, too_expensive: 0 },
      { price: 3.5, too_cheap: 0, cheap: 50.0, expensive: 0, too_expensive: 0 },
      { price: 4.0, too_cheap: 0, cheap: 0, expensive: 0, too_expensive: 0 },
      { price: 4.5, too_cheap: 0, cheap: 0, expensive: 0, too_expensive: 0 },
      { price: 5.0, too_cheap: 0, cheap: 0, expensive: 6.7, too_expensive: 0 },
      { price: 5.5, too_cheap: 0, cheap: 0, expensive: 6.7, too_expensive: 0 },
      { price: 6.0, too_cheap: 0, cheap: 0, expensive: 30.0, too_expensive: 0 },
      { price: 6.5, too_cheap: 0, cheap: 0, expensive: 73.3, too_expensive: 0 },
      { price: 7.0, too_cheap: 0, cheap: 0, expensive: 96.7, too_expensive: 0 },
      { price: 7.5, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 0 },
      { price: 8.0, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 23.3 },
      { price: 8.5, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 33.3 },
      { price: 9.0, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 73.3 },
      { price: 9.5, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 73.3 },
      { price: 10.0, too_cheap: 0, cheap: 0, expensive: 100, too_expensive: 100 },
    ],
    cohort_breakdown: [
      {
        cohort_name: 'Senior / Retiree',
        count: 7,
        population_share_pct: 23.33,
        mean_buy_intent_pct: 94.29,
        median_price_ceiling_sgd: 6.0,
      },
      {
        cohort_name: 'Working-Class',
        count: 10,
        population_share_pct: 33.33,
        mean_buy_intent_pct: 93.0,
        median_price_ceiling_sgd: 6.5,
      },
      {
        cohort_name: 'Suburban Family',
        count: 1,
        population_share_pct: 3.33,
        mean_buy_intent_pct: 100.0,
        median_price_ceiling_sgd: 6.5,
      },
      {
        cohort_name: 'Young Adult / Student',
        count: 4,
        population_share_pct: 13.33,
        mean_buy_intent_pct: 87.5,
        median_price_ceiling_sgd: 6.5,
      },
      {
        cohort_name: 'Affluent Private Property',
        count: 9,
        population_share_pct: 30.0,
        mean_buy_intent_pct: 88.89,
        median_price_ceiling_sgd: 6.5,
      },
    ],
  },
};
