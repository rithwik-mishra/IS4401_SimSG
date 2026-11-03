import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { ThumbsUp, Minus, ThumbsDown, TrendingUp } from "lucide-react";

export interface SentimentGaugeProps {
  meanBuyIntent: number;
  netSentimentScore: number;
  breakdown: {
    Positive: number;
    Neutral: number;
    Negative: number;
  };
  className?: string;
}

export function SentimentGauge({
  meanBuyIntent,
  netSentimentScore,
  breakdown,
  className,
}: SentimentGaugeProps) {
  const total = (breakdown.Positive || 0) + (breakdown.Neutral || 0) + (breakdown.Negative || 0);
  const posPct = total > 0 ? ((breakdown.Positive || 0) / total) * 100 : 0;
  const neuPct = total > 0 ? ((breakdown.Neutral || 0) / total) * 100 : 0;
  const negPct = total > 0 ? ((breakdown.Negative || 0) / total) * 100 : 0;

  const getNetScoreBadge = (score: number) => {
    if (score >= 0.5) return <span className="text-xs font-bold px-2.5 py-1 rounded-lg bg-[#edf4e8] text-[#4d9076] border border-[#dce7d2]">Strongly Favorable (+{score.toFixed(2)})</span>;
    if (score >= 0.1) return <span className="text-xs font-bold px-2.5 py-1 rounded-lg bg-[#edf4e8] text-[#4d9076] border border-[#dce7d2]">Moderately Positive (+{score.toFixed(2)})</span>;
    if (score >= -0.1) return <span className="text-xs font-bold px-2.5 py-1 rounded-lg bg-[#faf6e9] text-[#bda567] border border-[#eae2c8]">Neutral ({score.toFixed(2)})</span>;
    return <span className="text-xs font-bold px-2.5 py-1 rounded-lg bg-[#fbefeb] text-[#c87970] border border-[#eed7d0]">Unfavorable ({score.toFixed(2)})</span>;
  };

  return (
    <Card className={`border-[#e5e9e3] bg-white shadow-xs rounded-2xl ${className || ""}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#748078]">
              Audience Reaction
            </span>
            <CardTitle className="text-lg text-[#263c35]">Sentiment Breakdown</CardTitle>
            <CardDescription className="text-xs text-[#748078]">
              Aggregate cohort sentiment orientation across all respondents
            </CardDescription>
          </div>
          {getNetScoreBadge(netSentimentScore)}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* KPI Dual Display */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl bg-[#edf4e8]/60 border border-[#dce7d2] p-3.5">
            <div className="flex items-center gap-1.5 text-xs text-[#4d9076] font-semibold">
              <TrendingUp className="h-3.5 w-3.5" />
              <span>Mean Purchase Intent</span>
            </div>
            <div className="mt-1 text-2xl font-extrabold text-[#263c35] font-heading">
              {meanBuyIntent.toFixed(1)}%
            </div>
          </div>

          <div className="rounded-xl bg-[#fafbf8] border border-[#e5e9e3] p-3.5">
            <div className="flex items-center gap-1.5 text-xs text-[#748078] font-semibold">
              <span>Net Sentiment Score</span>
            </div>
            <div className="mt-1 text-2xl font-extrabold text-[#263c35] font-heading">
              {netSentimentScore > 0 ? `+${netSentimentScore.toFixed(2)}` : netSentimentScore.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Segmented Sentiment Bar */}
        <div className="space-y-2">
          <div className="h-3.5 w-full rounded-full overflow-hidden flex bg-[#f1f3eb]">
            <div
              style={{ width: `${posPct}%` }}
              className="bg-[#4d9076] transition-all duration-500"
              title={`Positive: ${posPct.toFixed(1)}%`}
            />
            <div
              style={{ width: `${neuPct}%` }}
              className="bg-[#bda567] transition-all duration-500"
              title={`Neutral: ${neuPct.toFixed(1)}%`}
            />
            <div
              style={{ width: `${negPct}%` }}
              className="bg-[#c87970] transition-all duration-500"
              title={`Negative: ${negPct.toFixed(1)}%`}
            />
          </div>

          <div className="flex items-center justify-between text-xs text-[#748078] pt-1">
            <div className="flex items-center gap-1.5">
              <ThumbsUp className="h-3.5 w-3.5 text-[#4d9076]" />
              <span className="font-bold text-[#263c35]">{breakdown.Positive || 0}</span> Positive ({posPct.toFixed(0)}%)
            </div>
            <div className="flex items-center gap-1.5">
              <Minus className="h-3.5 w-3.5 text-[#bda567]" />
              <span className="font-bold text-[#263c35]">{breakdown.Neutral || 0}</span> Neutral ({neuPct.toFixed(0)}%)
            </div>
            <div className="flex items-center gap-1.5">
              <ThumbsDown className="h-3.5 w-3.5 text-[#c87970]" />
              <span className="font-bold text-[#263c35]">{breakdown.Negative || 0}</span> Negative ({negPct.toFixed(0)}%)
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
