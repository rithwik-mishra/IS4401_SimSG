import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { TopNavigation } from "@/components/features/top-navigation";
import { PromptInput } from "@/components/ai/prompt-input";
import { Suggestions } from "@/components/ai/suggestions";
import { Reasoning, PipelineStep } from "@/components/ai/reasoning";
import { PersonaMessage } from "@/components/ai/message";
import { VerdictBanner } from "@/components/features/verdict-banner";
import { ObjectionClusters } from "@/components/features/objection-clusters";
import { PersonaDetailDialog } from "@/components/features/persona-detail-dialog";
import { PSMChart } from "@/components/figures/psm-chart";
import { CohortIntentChart } from "@/components/figures/cohort-intent-chart";
import { SentimentGauge } from "@/components/figures/sentiment-gauge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { PRESET_QUERIES, PLANNING_AREAS, PresetQuery } from "@/data/presets";
import { SAMPLE_SIMULATION_RESULT } from "@/data/sampleOutput";
import { runSimulation } from "@/services/api";
import { SimulateRequest, SimulateResponse, RepresentativePersona } from "@/types/api";
import {
  SlidersHorizontal,
  Users,
  MapPin,
  AlertCircle,
  FileText,
  UserCheck,
  LineChart as LineChartIcon,
  Filter,
} from "lucide-react";
import { TestingMenu, MAX_SEED } from "@/components/features/testing-menu";

export function App() {
  const [query, setQuery] = useState(
    "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"
  );
  const [overrideTown, setOverrideTown] = useState<string>("Ang Mo Kio");
  const [sampleSize, setSampleSize] = useState<number>(30);
  const [showOptions, setShowOptions] = useState<boolean>(false);
  const [cohortFilter, setCohortFilter] = useState<string>("all");
  // Testing only: optional seed; empty means "let the backend pick one".
  const [seedInput, setSeedInput] = useState<string>("");

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SimulateResponse | null>(SAMPLE_SIMULATION_RESULT);
  const [selectedPersona, setSelectedPersona] = useState<RepresentativePersona | null>(null);
  const [activeStatusHint, setActiveStatusHint] = useState<string>("");

  const activeRunIdRef = React.useRef<number>(0);

  // Multi-step reasoning pipeline configuration helper
  const getPipelineSteps = (size: number): PipelineStep[] => [
    {
      id: "decompose",
      title: "1. Natural Language Query Decomposition",
      description: "Extracting concept attributes, category norms, price baseline, and planning area.",
      status: "pending",
    },
    {
      id: "rag",
      title: "2. Layer 2 ChromaDB Qualitative RAG Retrieval",
      description: "Querying hyper-local qualitative context, dining habits, and neighborhood competition.",
      status: "pending",
    },
    {
      id: "sample",
      title: "3. Monte Carlo SingStat 2025 Demographic Sampling",
      description: `Sampling N=${size} statistically grounded personas from SingStat Census and HES 2023.`,
      status: "pending",
    },
    {
      id: "agents",
      title: "4. Async Batch Persona Evaluation Engine",
      description: "Executing concurrent LLM persona evaluations with Singlish reactions and price thresholds.",
      status: "pending",
    },
    {
      id: "psm",
      title: "5. Deterministic Van Westendorp PSM Econometrics",
      description: "Computing cumulative price curves, OPP, PMC, IPP, PME, and cohort likelihoods.",
      status: "pending",
    },
    {
      id: "synthesis",
      title: "6. Macro Synthesis & Strategic Pivot Agent",
      description: "Synthesizing executive summary, viability badge, objection clusters, and post-pivot uplift.",
      status: "pending",
    },
  ];

  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(() => getPipelineSteps(30));

  const handleSelectPreset = (preset: PresetQuery) => {
    setQuery(preset.query);
    setOverrideTown(preset.town);
  };

  const handleRunSimulation = async () => {
    if (!query.trim() || isLoading) return;

    // Track execution run to safely abandon outdated timer cycles
    const currentRunId = ++activeRunIdRef.current;
    const isCancelled = () => activeRunIdRef.current !== currentRunId;

    setIsLoading(true);
    setError(null);

    // Initial state: Step 0 is active with natural pacing
    const initialSteps = getPipelineSteps(sampleSize).map((s, idx) => ({
      ...s,
      status: idx === 0 ? ("running" as const) : ("pending" as const),
    }));
    setPipelineSteps(initialSteps);
    setActiveStatusHint("Extracting concept attributes & price baseline...");

    // Helper sleep function
    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      });

    // Start API request asynchronously
    const parsedSeed = seedInput.trim() === "" ? null : Number(seedInput);
    const requestPayload: SimulateRequest = {
      raw_query: query.trim(),
      override_town: overrideTown === "all" ? null : overrideTown,
      sample_size: sampleSize,
      seed: parsedSeed !== null && Number.isSafeInteger(parsedSeed) && parsedSeed <= MAX_SEED ? parsedSeed : null,
      use_mock: false,
    };

    let apiResult: SimulateResponse | null = null;
    let apiError: any = null;
    let apiCompleted = false;

    const apiPromise = runSimulation(requestPayload, {
      allowMockFallback: true,
    })
      .then((data) => {
        apiResult = data;
        apiCompleted = true;
      })
      .catch((err) => {
        apiError = err;
        apiCompleted = true;
      });

    // Higher, human-readable delays between steps (~1.8s - 2.4s per step)
    const stepConfigs = [
      {
        delay: 1800,
        hint: "Extracting concept attributes & category norms...",
      },
      {
        delay: 2000,
        hint: "Querying ChromaDB Singapore qualitative database...",
      },
      {
        delay: 1900,
        hint: `Sampling N=${sampleSize} stratified SingStat resident profiles...`,
      },
      {
        delay: 2400,
        hint: `Simulating ${sampleSize} resident focus group reactions & price acceptance...`,
      },
      {
        delay: 1900,
        hint: "Calculating Van Westendorp curves (OPP, PMC, IPP, PME)...",
      },
      {
        delay: 2200,
        hint: "Synthesizing executive consensus & strategic recommendations...",
      },
    ];

    const synthesisFollowUpHints = [
      "Aggregating objection clusters & price sensitivity frontiers...",
      "Formulating actionable pivots & demographic uplift...",
      "Finalizing focus group simulation report & charts...",
    ];

    try {
      // Step through stages 0 to 4 with higher, readable delays
      for (let stepIdx = 0; stepIdx < 5; stepIdx++) {
        if (isCancelled()) return;
        if (apiError) throw apiError;

        // If backend has completed, accelerate remaining steps smoothly
        const delay = apiCompleted ? 400 : stepConfigs[stepIdx].delay;
        await sleep(delay);

        if (isCancelled()) return;
        if (apiError) throw apiError;

        const nextStepIdx = stepIdx + 1;
        setPipelineSteps((prev) =>
          prev.map((s, idx) => ({
            ...s,
            status:
              idx < nextStepIdx
                ? "completed"
                : idx === nextStepIdx
                  ? "running"
                  : "pending",
          }))
        );
        setActiveStatusHint(stepConfigs[nextStepIdx].hint);
      }

      // Stage 6 (Macro Synthesis & Strategic Pivot Agent)
      if (isCancelled()) return;
      if (apiError) throw apiError;

      // Ensure stage 6 runs with active progress feedback
      let hintIndex = 0;
      const stage6StartTime = Date.now();
      const minStage6Duration = apiCompleted ? 600 : 2000;

      while (!apiCompleted || Date.now() - stage6StartTime < minStage6Duration) {
        if (isCancelled()) return;
        if (apiError) throw apiError;

        await sleep(800);

        if (isCancelled()) return;
        if (apiError) throw apiError;

        // If waiting for backend, cycle through dynamic progress messages so it never feels stuck
        if (!apiCompleted && Date.now() - stage6StartTime > 2200) {
          const nextHint =
            synthesisFollowUpHints[hintIndex % synthesisFollowUpHints.length];
          setActiveStatusHint(nextHint);
          hintIndex++;
        }
      }

      // Await final API response
      await apiPromise;
      if (isCancelled()) return;
      if (apiError) throw apiError;
      if (!apiResult) throw new Error("Simulation completed without return data.");

      // Mark all 6 stages completed (100% complete state)
      setPipelineSteps((prev) => prev.map((s) => ({ ...s, status: "completed" })));
      setActiveStatusHint("Simulation complete! Launching analysis workspace...");

      // Hold completion state for 750ms so user can see all green checks and 100%
      await sleep(750);
      if (isCancelled()) return;

      setResults(apiResult);
      setIsLoading(false);
      setActiveStatusHint("");
    } catch (err: any) {
      if (isCancelled()) return;
      setError(err.message || "An unexpected error occurred during simulation.");
      setIsLoading(false);
      setActiveStatusHint("");
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
  };

  // Automatically detect candidate price from user pitch for reference line
  const priceMatch = query.match(/S?\$([0-9]+(?:\.[0-9]{1,2})?)/i);
  const candidatePrice = priceMatch ? parseFloat(priceMatch[1]) : null;

  // Filter resident voices by cohort if selected
  const displayedPersonas = results
    ? results.personas.filter(
      (p) => cohortFilter === "all" || p.demographic_cohort === cohortFilter
    )
    : [];

  const availableCohorts = results
    ? Array.from(new Set(results.personas.map((p) => p.demographic_cohort)))
    : [];

  return (
    <div className="min-h-screen bg-[#f5f6f2] text-[#263c35] flex flex-col font-sans">
      {/* Prominent Navbar - No blur, rich forest green, no pin, no mock mode */}
      <TopNavigation
        onReset={handleReset}
        hasResults={!!results}
        rightSlot={
          <TestingMenu
            seedInput={seedInput}
            onSeedInputChange={setSeedInput}
            lastSeed={results?.seed}
            disabled={isLoading}
          />
        }
      />

      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 py-6 sm:py-8 space-y-8">
        {/* Hero Conversational Prompt Section */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-4 max-w-4xl mx-auto"
        >
          <div className="text-center space-y-2">
            <span className="text-[11px] font-bold tracking-[0.12em] text-[#748078] uppercase">
              Singapore Micro-SME Market Research
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#263c35] font-heading">
              What are you thinking of opening?
            </h1>
            <p className="text-sm text-[#748078] max-w-2xl mx-auto leading-relaxed">
              Describe your business concept and hear authentic reactions from <strong>{sampleSize} SingStat census-grounded resident personas</strong>, with econometric price sensitivity curves and strategic pivot guidance.
            </p>
          </div>

          {/* Prompt Input with Shadcn AI controls */}
          <PromptInput
            value={query}
            onChange={setQuery}
            onSubmit={handleRunSimulation}
            isLoading={isLoading}
            placeholder="A S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets…"
          >
            {/* Planning Area Selector with No Emojis */}
            <div className="flex items-center gap-1.5">
              <MapPin className="h-4 w-4 text-[#214f43]" />
              <Select value={overrideTown} onValueChange={setOverrideTown}>
                <SelectTrigger className="h-9 text-xs sm:text-sm font-semibold w-[170px] sm:w-[200px] bg-white border-[#e0e6dd] text-[#263c35]">
                  <SelectValue placeholder="Location" />
                </SelectTrigger>
                <SelectContent>
                  {PLANNING_AREAS.map((area) => (
                    <SelectItem key={area.value} value={area.value} className="text-xs sm:text-sm">
                      {area.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Toggle sample size options with motion feedback */}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setShowOptions(!showOptions)}
              className="h-9 px-3 text-xs text-[#748078] hover:text-[#263c35] hover:bg-[#e8efdf]/50"
            >
              <SlidersHorizontal className="h-3.5 w-3.5 mr-1 text-[#748078]" />
              <span>N={sampleSize} respondents</span>
            </Button>
          </PromptInput>

          {/* Collapsible Advanced Parameters with Motion.dev */}
          <AnimatePresence>
            {showOptions && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="rounded-2xl border border-[#e5e9e3] bg-white p-5 space-y-3.5 shadow-sm">
                  <div className="flex items-center justify-between text-xs font-medium">
                    <span className="flex items-center gap-1.5 text-[#748078]">
                      <Users className="h-4 w-4 text-[#214f43]" /> Persona Sample Size:
                      <strong className="text-[#263c35] ml-1 font-heading text-sm">{sampleSize} residents</strong>
                    </span>
                    <span className="text-[11px] text-[#748078]">
                      Standard Sample: 30 (Adjust between 5 and 100)
                    </span>
                  </div>
                  {/* High-visibility, tactile slider */}
                  <Slider
                    value={[sampleSize]}
                    min={5}
                    max={100}
                    step={5}
                    onValueChange={(vals) => setSampleSize(vals[0])}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Quick Singapore Scenario Presets */}
          <Suggestions suggestions={PRESET_QUERIES} onSelect={handleSelectPreset} />
        </motion.div>

        {/* Live Simulation Reasoning Tracker */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="max-w-4xl mx-auto pt-2"
            >
              <Reasoning
                steps={pipelineSteps}
                currentStepIndex={pipelineSteps.findIndex((s) => s.status === "running")}
                activeStatusHint={activeStatusHint}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Alert */}
        {error && (
          <div className="max-w-4xl mx-auto rounded-xl border border-[#c87970]/40 bg-[#fff3f0] p-4 text-xs text-[#8b423b] flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-[#c87970] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold text-sm block">Simulation Error</span>
              <p>{error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRunSimulation}
                className="mt-2 text-xs border-[#c87970]/30 hover:bg-[#c87970]/10 text-[#8b423b]"
              >
                Retry Simulation
              </Button>
            </div>
          </div>
        )}

        {/* Simulation Results Workspace */}
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="space-y-6 pt-2"
          >
            <Tabs defaultValue="synthesis" className="w-full">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#e5e9e3] pb-3">
                <TabsList className="grid grid-cols-3 w-full sm:w-auto h-10 bg-[#e8efdf]/50 p-1">
                  <TabsTrigger value="synthesis" className="flex items-center gap-1.5 text-xs sm:text-sm data-[state=active]:bg-white data-[state=active]:text-[#214f43]">
                    <FileText className="h-3.5 w-3.5" />
                    <span>Strategic Synthesis</span>
                  </TabsTrigger>
                  <TabsTrigger value="personas" className="flex items-center gap-1.5 text-xs sm:text-sm data-[state=active]:bg-white data-[state=active]:text-[#214f43]">
                    <UserCheck className="h-3.5 w-3.5" />
                    <span>Resident Voices ({results.personas.length})</span>
                  </TabsTrigger>
                  <TabsTrigger value="econometrics" className="flex items-center gap-1.5 text-xs sm:text-sm data-[state=active]:bg-white data-[state=active]:text-[#214f43]">
                    <LineChartIcon className="h-3.5 w-3.5" />
                    <span>Price Sensitivity & PSM</span>
                  </TabsTrigger>
                </TabsList>

                <div className="flex items-center gap-2 text-xs text-[#748078]">
                  <span className="px-2.5 py-1 rounded-md bg-white border border-[#e5e9e3] font-medium">
                    Total Residents: {results.personas.length}
                  </span>
                  <span className="px-2.5 py-1 rounded-md bg-white border border-[#e5e9e3] font-medium">
                    Target: {overrideTown === "all" ? "Singapore-wide" : overrideTown}
                  </span>
                </div>
              </div>

              {/* TAB 1: Strategic Synthesis */}
              <TabsContent value="synthesis" className="space-y-6 pt-4">
                <VerdictBanner synthesis={results.recommendations} />
                <ObjectionClusters objections={results.recommendations.clustered_objections} />
              </TabsContent>

              {/* TAB 2: Resident Personas (Showing All N Results) */}
              <TabsContent value="personas" className="space-y-5 pt-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h3 className="text-base sm:text-lg font-bold text-[#263c35] font-heading">
                      Simulated Resident Focus Group (All {results.personas.length} Respondents)
                    </h3>
                    <p className="text-xs text-[#748078]">
                      Authentic local perspectives, price ceilings, and objections drawn from SingStat Census demographics
                    </p>
                  </div>
                  <span className="text-xs text-[#748078] bg-[#f8faf5] px-3 py-1 rounded-lg border border-[#e5e9e3]">
                    Click any card to inspect full spending breakdown
                  </span>
                </div>

                {/* Cohort Filter Chips */}
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  <span className="text-xs text-[#748078] flex items-center gap-1 mr-1">
                    <Filter className="h-3.5 w-3.5" /> Filter:
                  </span>
                  <button
                    type="button"
                    onClick={() => setCohortFilter("all")}
                    className={`text-xs font-semibold px-3 py-1 rounded-lg transition-all cursor-pointer ${cohortFilter === "all"
                        ? "bg-[#214f43] text-white shadow-xs"
                        : "bg-white text-[#748078] border border-[#e5e9e3] hover:border-[#214f43]/40"
                      }`}
                  >
                    All Cohorts ({results.personas.length})
                  </button>
                  {availableCohorts.map((cohort) => {
                    const count = results.personas.filter((p) => p.demographic_cohort === cohort).length;
                    return (
                      <button
                        key={cohort}
                        type="button"
                        onClick={() => setCohortFilter(cohort)}
                        className={`text-xs font-medium px-3 py-1 rounded-lg transition-all cursor-pointer ${cohortFilter === cohort
                            ? "bg-[#214f43] text-white shadow-xs font-semibold"
                            : "bg-white text-[#748078] border border-[#e5e9e3] hover:border-[#214f43]/40"
                          }`}
                      >
                        {cohort} ({count})
                      </button>
                    );
                  })}
                </div>

                {/* Responsive Grid of All N Resident Persona Cards with Motion.dev Animations */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-1">
                  <AnimatePresence mode="popLayout">
                    {displayedPersonas.map((persona, idx) => (
                      <motion.div
                        key={persona.persona_id}
                        layout
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.25, delay: Math.min(idx * 0.02, 0.35) }}
                        whileHover={{ y: -3, transition: { duration: 0.15 } }}
                      >
                        <PersonaMessage
                          persona={persona}
                          onClickDetails={setSelectedPersona}
                        />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </TabsContent>

              {/* TAB 3: Econometrics & PSM */}
              <TabsContent value="econometrics" className="space-y-6 pt-4">
                {/* 4-Curve Van Westendorp Price Sensitivity Chart */}
                <PSMChart
                  curves={results.statistics.psm_curves}
                  intersections={results.statistics.psm_intersections}
                  candidatePrice={candidatePrice}
                />

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Cohort Likelihood-to-Buy */}
                  <CohortIntentChart cohorts={results.statistics.cohort_breakdown} />

                  {/* Net Sentiment Gauge */}
                  <SentimentGauge
                    meanBuyIntent={results.statistics.mean_buy_intent_pct}
                    netSentimentScore={results.statistics.net_sentiment_score}
                    breakdown={results.statistics.sentiment_breakdown}
                  />
                </div>
              </TabsContent>
            </Tabs>
          </motion.div>
        )}
      </main>

      {/* Persona Detail Modal */}
      <PersonaDetailDialog
        persona={selectedPersona}
        isOpen={!!selectedPersona}
        onClose={() => setSelectedPersona(null)}
      />

      {/* Footer matching SimSG warmth */}
      <footer className="mt-auto border-t border-[#e5e9e3] bg-white py-5 text-center text-xs text-[#748078]">
        <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span className="font-heading font-extrabold text-[#214f43]">SimSG.</span>
          <span>SingStat 2025 Census & HES 2023 Expenditure Benchmarks • ChromaDB Layer 2 RAG</span>
          <span>FastAPI + React 19</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
