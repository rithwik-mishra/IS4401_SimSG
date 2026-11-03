import React from "react";
import { MacroSynthesisReport } from "@/types/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { CheckCircle2, AlertTriangle, XCircle, ArrowUpRight, Compass } from "lucide-react";

export interface VerdictBannerProps {
  synthesis: MacroSynthesisReport;
  className?: string;
}

export function VerdictBanner({ synthesis, className }: VerdictBannerProps) {
  const getVerdictDetails = (badge: string) => {
    const text = badge.toUpperCase();
    if (text.includes("VIABLE") && !text.includes("RISK")) {
      return {
        icon: <CheckCircle2 className="h-6 w-6 text-[#4d9076] shrink-0" />,
        bgColor: "bg-[#edf4e8] border-[#dce7d2]",
        symbolBg: "bg-[#dceacb] text-[#4d9076]",
        label: "VIABLE",
      };
    }
    if (text.includes("HIGH RISK")) {
      return {
        icon: <XCircle className="h-6 w-6 text-[#c87970] shrink-0" />,
        bgColor: "bg-[#fbefeb] border-[#eed7d0]",
        symbolBg: "bg-[#f3dcd5] text-[#c87970]",
        label: "HIGH RISK — PIVOT REQUIRED",
      };
    }
    return {
      icon: <AlertTriangle className="h-6 w-6 text-[#bda567] shrink-0" />,
      bgColor: "bg-[#faf6e9] border-[#eae2c8]",
      symbolBg: "bg-[#efe6c4] text-[#bda567]",
      label: "MODERATE RISK",
    };
  };

  const verdict = getVerdictDetails(synthesis.verdict_badge);

  return (
    <div className={`space-y-4 ${className || ""}`}>
      {/* Viability Status Banner */}
      <div className={`rounded-2xl border p-5 sm:p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${verdict.bgColor} shadow-xs`}>
        <div className="flex items-center gap-3.5">
          <div className={`w-12 h-12 rounded-full flex items-center justify-center ${verdict.symbolBg}`}>
            {verdict.icon}
          </div>
          <div>
            <span className="text-[11px] font-bold tracking-[0.14em] uppercase text-[#748078] block">
              Simulation Verdict
            </span>
            <span className="text-xl sm:text-2xl font-extrabold tracking-tight text-[#263c35] font-heading">
              {synthesis.verdict_badge}
            </span>
          </div>
        </div>

        {synthesis.simulated_post_pivot_uplift > 0 && (
          <div className="flex items-center gap-2.5 rounded-xl bg-white px-4 py-2 shadow-xs border border-[#e5e9e3]">
            <ArrowUpRight className="h-4 w-4 text-[#4d9076]" />
            <span className="text-xs text-[#748078] font-medium">Post-Pivot Uplift:</span>
            <span className="text-base font-extrabold text-[#4d9076] font-heading">
              +{synthesis.simulated_post_pivot_uplift.toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      {/* Executive Summary Card */}
      <Card className="border-[#e5e9e3] bg-white shadow-xs rounded-2xl">
        <CardHeader className="pb-2">
          <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#748078]">
            What the Responses Say
          </span>
          <CardTitle className="text-lg text-[#263c35]">Executive Market Synthesis</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm sm:text-[15px] text-[#263c35] leading-relaxed">
            {synthesis.executive_summary}
          </p>
        </CardContent>
      </Card>

      {/* Recommended Strategic Pivot Card */}
      {synthesis.recommended_pivot && (
        <Card className="border-[#dce7d2] bg-[#f8faf5] shadow-xs rounded-2xl">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 text-[#214f43]">
              <Compass className="h-4 w-4 text-[#214f43]" />
              <CardTitle className="text-base text-[#214f43]">
                Suggested Operational Pivot & Action Plan
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm sm:text-[15px] text-[#263c35] leading-relaxed font-medium">
              {synthesis.recommended_pivot}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
