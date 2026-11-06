import React from "react";
import { RepresentativePersona } from "@/types/api";
import { formatIncomeBand, formatSGD } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  User,
  Building2,
  Wallet,
  DollarSign,
  MessageSquareQuote,
  AlertCircle,
  TrendingUp,
  Briefcase,
  Home,
} from "lucide-react";

export interface PersonaDetailDialogProps {
  persona: RepresentativePersona | null;
  isOpen: boolean;
  onClose: () => void;
}

export function PersonaDetailDialog({
  persona,
  isOpen,
  onClose,
}: PersonaDetailDialogProps) {
  if (!persona) return null;

  const getCohortColor = (cohort: string) => {
    switch (cohort) {
      case "Senior / Retiree":
        return "#8c79b4";
      case "Working-Class":
        return "#6b9a90";
      case "Suburban Family":
        return "#d6ad69";
      case "Young Adult / Student":
        return "#8fa861";
      case "Affluent Private Property":
        return "#7899bf";
      default:
        return "#214f43";
    }
  };

  const cohortColor = getCohortColor(persona.demographic_cohort);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md sm:max-w-xl bg-white text-[#263c35] border border-[#e5e9e3] shadow-2xl p-6 sm:p-7 rounded-2xl">
        <DialogHeader className="pb-2">
          <div className="flex flex-wrap items-center justify-between gap-2 pr-6">
            <div className="flex items-center gap-2.5">
              <DialogTitle className="text-xl font-bold text-[#263c35] font-heading">
                Respondent: {persona.persona_id}
              </DialogTitle>
              <span
                style={{ backgroundColor: `${cohortColor}20`, color: cohortColor }}
                className="text-xs font-bold px-2.5 py-0.5 rounded-lg border border-[#e5e9e3]"
              >
                {persona.demographic_cohort}
              </span>
            </div>
          </div>
          <DialogDescription className="text-xs text-[#748078]">
            SingStat 2025 Census Profile & HES 2023 Expenditure Benchmarks
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-1">
          {/* Reaction Quote Highlight */}
          <div className="rounded-xl bg-[#f8faf5] border-l-4 border-[#214f43] p-4 text-sm sm:text-[15px] text-[#263c35] italic leading-relaxed shadow-xs">
            <MessageSquareQuote className="h-4 w-4 inline mr-2 text-[#4d9076] not-italic" />
            "{persona.quote}"
          </div>

          {/* Demographic Metrics Grid */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-xl border border-[#e5e9e3] bg-[#fafbf8] p-3.5 space-y-1 shadow-xs">
              <span className="text-[#748078] flex items-center gap-1.5 font-medium">
                <User className="h-3.5 w-3.5 text-[#214f43]" /> Demographics
              </span>
              <p className="font-extrabold text-[#263c35] text-sm">
                {persona.age} years • {persona.gender}
              </p>
              <p className="text-[#748078] flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {persona.dwelling_type} Flat
              </p>
              {/* Ages 60+ (labour force) and 65+ (living arrangement) only; 2025 national data */}
              {persona.in_labour_force != null && (
                <p className="text-[#748078] flex items-center gap-1" data-testid="persona-labour-force">
                  <Briefcase className="h-3 w-3" />
                  {persona.in_labour_force ? "In the labour force" : "Not in the labour force"}
                </p>
              )}
              {persona.living_arrangement && (
                <p className="text-[#748078] flex items-start gap-1" data-testid="persona-living-arrangement">
                  <Home className="h-3 w-3 mt-0.5 shrink-0" />
                  <span>{persona.living_arrangement.charAt(0).toUpperCase() + persona.living_arrangement.slice(1)}</span>
                </p>
              )}
            </div>

            <div className="rounded-xl border border-[#e5e9e3] bg-[#fafbf8] p-3.5 space-y-1 shadow-xs">
              <span className="text-[#748078] flex items-center gap-1.5 font-medium">
                <Wallet className="h-3.5 w-3.5 text-[#214f43]" /> Household Income (from work)
              </span>
              <p className="font-extrabold text-[#263c35] text-sm">{formatIncomeBand(persona)} /mo</p>
              <p className="text-[11px] leading-snug text-[#748078]">
                Whole household; excludes pensions, CPF payouts, rental income and family support
              </p>
              <p className="text-[#748078] flex items-center gap-1">
                <DollarSign className="h-3 w-3" />
                Food: {formatSGD(persona.daily_dining_budget_sgd)}/day
              </p>
            </div>
          </div>

          {/* Purchase Probability & Sentiment */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-xl border border-[#e5e9e3] bg-white p-3.5 space-y-1 shadow-xs">
              <span className="text-[#748078] flex items-center gap-1.5 font-medium">
                <TrendingUp className="h-3.5 w-3.5 text-[#4d9076]" /> Purchase Likelihood
              </span>
              <p className="font-extrabold text-[#263c35] text-lg font-heading">
                {persona.buy_probability_pct}%
              </p>
            </div>

            <div className="rounded-xl border border-[#e5e9e3] bg-white p-3.5 space-y-1 shadow-xs">
              <span className="text-[#748078] font-medium block">Sentiment Reaction</span>
              <p className="font-extrabold text-[#263c35] text-lg font-heading">
                {persona.sentiment}
              </p>
            </div>
          </div>

          {/* Van Westendorp Pricing Boundaries */}
          <div className="rounded-xl border border-[#e5e9e3] bg-[#fafbf8] p-4 space-y-2.5 shadow-xs">
            <span className="text-xs font-bold text-[#263c35] block">
              Van Westendorp Price Sensitivity Perceptions (SGD)
            </span>
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="rounded-lg bg-white border border-[#e5e9e3] p-2.5">
                <span className="text-[10px] text-[#748078] block">Too Cheap</span>
                <span className="font-bold text-[#8c79b4] text-sm">{formatSGD(persona.too_cheap)}</span>
              </div>
              <div className="rounded-lg bg-white border border-[#e5e9e3] p-2.5">
                <span className="text-[10px] text-[#748078] block">Bargain</span>
                <span className="font-bold text-[#4d9076] text-sm">{formatSGD(persona.cheap)}</span>
              </div>
              <div className="rounded-lg bg-white border border-[#e5e9e3] p-2.5">
                <span className="text-[10px] text-[#748078] block">Expensive</span>
                <span className="font-bold text-[#d6ad69] text-sm">{formatSGD(persona.expensive)}</span>
              </div>
              <div className="rounded-lg bg-white border border-[#e5e9e3] p-2.5">
                <span className="text-[10px] text-[#748078] block">Too Exp</span>
                <span className="font-bold text-[#c87970] text-sm">{formatSGD(persona.too_expensive)}</span>
              </div>
            </div>
          </div>

          {/* Primary Objection */}
          {persona.primary_objection && (
            <div className="rounded-xl border border-[#eae2c8] bg-[#faf6e9] p-3.5 text-xs flex items-start gap-2.5 shadow-xs">
              <AlertCircle className="h-4 w-4 text-[#bda567] shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-[#263c35]">
                  Primary Hesitation Point:
                </span>
                <p className="text-[#657658] mt-0.5 leading-relaxed font-medium">
                  {persona.primary_objection}
                </p>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
