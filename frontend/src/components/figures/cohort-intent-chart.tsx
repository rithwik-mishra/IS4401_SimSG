import React from "react";
import { CohortBreakdown } from "@/types/api";
import { formatSGD } from "@/lib/utils";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

export interface CohortIntentChartProps {
  cohorts: CohortBreakdown[];
  className?: string;
}

export function CohortIntentChart({ cohorts, className }: CohortIntentChartProps) {
  const getCohortColor = (name: string) => {
    switch (name) {
      case "Senior / Retiree":
        return "bg-[#8c79b4]";
      case "Working-Class":
        return "bg-[#6b9a90]";
      case "Suburban Family":
        return "bg-[#d6ad69]";
      case "Young Adult / Student":
        return "bg-[#8fa861]";
      case "Affluent Private Property":
        return "bg-[#7899bf]";
      default:
        return "bg-[#214f43]";
    }
  };

  const getCohortDotColor = (name: string) => {
    switch (name) {
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

  return (
    <Card className={`border-[#e5e9e3] bg-white shadow-xs rounded-2xl ${className || ""}`}>
      <CardHeader className="pb-3">
        <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#748078]">
          Demographic Purchase Intent
        </span>
        <CardTitle className="text-lg text-[#263c35]">Likelihood to Buy by Segment</CardTitle>
        <CardDescription className="text-xs text-[#748078]">
          Purchase probabilities and median price ceilings across SingStat population cohorts
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {cohorts.map((cohort) => (
          <div key={cohort.cohort_name} className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span
                  style={{ backgroundColor: getCohortDotColor(cohort.cohort_name) }}
                  className="h-2.5 w-2.5 rounded-full shrink-0"
                />
                <span className="font-bold text-[#263c35]">
                  {cohort.cohort_name}
                </span>
                <span className="text-[11px] text-[#748078]">
                  ({cohort.count} reps • {cohort.population_share_pct.toFixed(0)}%)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium text-[#748078] bg-[#f8faf5] px-2 py-0.5 rounded border border-[#e5e9e3]">
                  Ceiling: {formatSGD(cohort.median_price_ceiling_sgd)}
                </span>
                <span className="font-bold text-[#263c35] min-w-[3rem] text-right">
                  {cohort.mean_buy_intent_pct.toFixed(1)}%
                </span>
              </div>
            </div>

            <Progress
              value={cohort.mean_buy_intent_pct}
              className="h-2.5 bg-[#f1f3eb]"
              indicatorClassName={getCohortColor(cohort.cohort_name)}
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
