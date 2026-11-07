import React from "react";
import { formatIncomeBand, formatSGD, getCohortColor } from "@/lib/utils";
import { RepresentativePersona } from "@/types/api";
import { MessageSquareQuote, DollarSign, Wallet, Building2, AlertCircle } from "lucide-react";

export interface PersonaMessageProps {
  persona: RepresentativePersona;
  onClickDetails?: (persona: RepresentativePersona) => void;
  className?: string;
}

export function PersonaMessage({
  persona,
  onClickDetails,
  className,
}: PersonaMessageProps) {
  const getSentimentDetails = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case "positive":
        return { text: "Positive", bg: "bg-[#edf4e8]", color: "text-[#4d9076]", border: "border-[#dce7d2]" };
      case "negative":
        return { text: "Negative", bg: "bg-[#fbefeb]", color: "text-[#c87970]", border: "border-[#eed7d0]" };
      default:
        return { text: "Neutral", bg: "bg-[#faf6e9]", color: "text-[#bda567]", border: "border-[#eae2c8]" };
    }
  };

  const cohortColor = getCohortColor(persona.demographic_cohort);
  const sent = getSentimentDetails(persona.sentiment);

  const getInitials = (cohort: string) => {
    const parts = cohort.split(" ");
    if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`;
    return cohort.slice(0, 2).toUpperCase();
  };

  return (
    <div
      onClick={() => onClickDetails && onClickDetails(persona)}
      className={`group relative rounded-2xl border border-[#e4e9e0] bg-white p-5 shadow-xs transition-all hover:border-[#214f43]/40 hover:shadow-md cursor-pointer ${
        className || ""
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={`h-11 w-11 rounded-xl flex items-center justify-center font-bold text-sm group-hover:scale-105 transition-transform border ${cohortColor}`}
          >
            {getInitials(persona.demographic_cohort)}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm text-[#263c35]">
                {persona.persona_id}
              </span>
              <span className="text-xs text-[#748078]">
                {persona.age}y • {persona.gender}
              </span>
            </div>
            <div className="mt-1">
              <span
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-md inline-block border ${cohortColor}`}
              >
                {persona.demographic_cohort}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${sent.bg} ${sent.color} ${sent.border}`}>
            {sent.text}
          </span>
          <span className="text-[11px] font-bold text-[#263c35]">
            {persona.buy_probability_pct}% intent
          </span>
        </div>
      </div>

      {/* Reaction Quote Bubble */}
      <div className="mt-4 relative rounded-xl bg-[#fafbf8] p-3.5 text-xs sm:text-[13px] text-[#263c35] italic border-l-2 border-[#b7cda5]">
        <MessageSquareQuote className="h-4 w-4 inline mr-1 text-[#b7cda5] not-italic" />
        "{persona.quote}"
      </div>

      {/* Primary Concern / Objection */}
      {persona.primary_objection && (
        <div className="mt-3 flex items-start gap-1.5 text-xs text-[#748078]">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 text-[#bda567] mt-0.5" />
          <span className="line-clamp-2 leading-relaxed">
            <strong className="text-[#263c35] font-semibold">Concern:</strong> {persona.primary_objection}
          </span>
        </div>
      )}

      {/* Demographics & Budget Chips */}
      <div className="mt-3.5 pt-3 border-t border-[#f0f3ee] grid grid-cols-3 gap-2 text-[11px] text-[#748078]">
        <div className="flex items-center gap-1">
          <Building2 className="h-3 w-3 text-[#748078]/70" />
          <span className="truncate">{persona.dwelling_type}</span>
        </div>
        <div className="flex items-center gap-1">
          <Wallet className="h-3 w-3 text-[#748078]/70" />
          <span className="truncate" title={`Household income from work: ${formatIncomeBand(persona)} per month`}>
            {formatIncomeBand(persona, true)}/mo
          </span>
        </div>
        <div className="flex items-center gap-1">
          <DollarSign className="h-3 w-3 text-[#748078]/70" />
          <span className="truncate">{formatSGD(persona.daily_dining_budget_sgd)}/day</span>
        </div>
      </div>

      {/* Price Sensitivity Markers */}
      <div className="mt-3 flex items-center justify-between text-[11px] text-[#748078] bg-[#f8faf5] px-2.5 py-1.5 rounded-lg border border-[#e5e9e3]">
        <span>Cheap: {formatSGD(persona.cheap)}</span>
        <span className="font-bold text-[#214f43]">
          Ceiling: {formatSGD(persona.price_ceiling_sgd)}
        </span>
        <span>Too Exp: {formatSGD(persona.too_expensive)}</span>
      </div>
    </div>
  );
}
