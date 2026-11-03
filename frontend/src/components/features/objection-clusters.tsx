import React from "react";
import { ClusteredObjection } from "@/types/api";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MessageSquareQuote, ShieldAlert } from "lucide-react";

export interface ObjectionClustersProps {
  objections: ClusteredObjection[];
  className?: string;
}

export function ObjectionClusters({ objections, className }: ObjectionClustersProps) {
  return (
    <Card className={`border-[#e5e9e3] bg-white shadow-xs rounded-2xl ${className || ""}`}>
      <CardHeader className="pb-3">
        <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#748078]">
          What Might Hold People Back
        </span>
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-[#bda567]" />
          <CardTitle className="text-lg text-[#263c35]">Main Concerns & Objections</CardTitle>
        </div>
        <CardDescription className="text-xs text-[#748078]">
          Consumer hesitation themes synthesized across qualitative respondent evaluations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {objections.map((cluster, index) => (
          <div
            key={`cluster-${index}`}
            className="rounded-xl border border-[#e5e9e3] bg-[#fafbf8] p-4 space-y-3 transition-all hover:bg-[#f6f8f4]"
          >
            <div className="flex items-center justify-between text-xs sm:text-sm">
              <span className="inline-block rounded-lg px-3 py-1 bg-[#f0f3e9] text-[#697d54] font-bold">
                {cluster.theme}
              </span>
              <div className="flex items-center gap-2.5">
                <Progress
                  value={cluster.frequency_pct}
                  className="w-24 h-2 bg-[#edf0e7]"
                  indicatorClassName="bg-[#a4b78d]"
                />
                <span className="text-xs font-bold text-[#657658] min-w-[3.5rem] text-right">
                  {cluster.frequency_pct}% share
                </span>
              </div>
            </div>

            {/* Quotes List */}
            {cluster.representative_quotes && cluster.representative_quotes.length > 0 && (
              <div className="space-y-2 pt-1">
                {cluster.representative_quotes.map((quote, qIdx) => (
                  <div
                    key={`quote-${qIdx}`}
                    className="flex items-start gap-2.5 text-xs text-[#7c8676] italic rounded-lg bg-white p-3 border border-[#e4eadd]"
                  >
                    <MessageSquareQuote className="h-4 w-4 shrink-0 text-[#b9c4ab] not-italic mt-0.5" />
                    <span className="leading-relaxed">"{quote}"</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
