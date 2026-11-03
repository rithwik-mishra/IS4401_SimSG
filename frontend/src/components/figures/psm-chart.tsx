import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import { PSMCurvePoint, PSMIntersections } from "@/types/api";
import { formatSGD } from "@/lib/utils";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Info, HelpCircle } from "lucide-react";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface PSMChartProps {
  curves: PSMCurvePoint[];
  intersections: PSMIntersections;
  candidatePrice?: number | null;
  className?: string;
}

export function PSMChart({
  curves,
  intersections,
  candidatePrice,
  className,
}: PSMChartProps) {
  const { pmc, opp, ipp, pme } = intersections;

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="rounded-xl border border-[#e5e9e3] bg-white p-3.5 shadow-lg text-xs">
          <p className="font-bold text-[#263c35] mb-2 pb-1.5 border-b border-[#f0f3ee]">
            Price Point: {formatSGD(Number(label))}
          </p>
          <div className="space-y-1.5">
            {payload.map((entry: any, index: number) => (
              <div key={`item-${index}`} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 font-medium" style={{ color: entry.color }}>
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
                  {entry.name}:
                </span>
                <span className="font-bold text-[#263c35]">{entry.value.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className={`border-[#e5e9e3] bg-white shadow-xs rounded-2xl ${className || ""}`}>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="text-[10px] font-bold tracking-[0.14em] uppercase text-[#748078]">
              Finding a Price that Fits
            </span>
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg text-[#263c35]">
                Van Westendorp Price Sensitivity Meter
              </CardTitle>
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <button type="button" className="text-[#748078] hover:text-[#263c35] cursor-pointer">
                      <HelpCircle className="h-4 w-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-[#214f43] text-white">
                    Plots 4 cumulative price perception curves to identify the Optimal Price Point (OPP) and Acceptable Price Range (PMC to PME).
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
            </div>
            <CardDescription className="text-xs text-[#748078]">
              Cumulative resident price perceptions (Prices in Singapore Dollars)
            </CardDescription>
          </div>

          {/* Pricing Markers Summary Badges */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold px-3 py-1 rounded-lg bg-[#edf4e8] text-[#4d9076] border border-[#dce7d2]">
              OPP: {formatSGD(opp)}
            </span>
            <span className="text-xs font-medium px-3 py-1 rounded-lg bg-[#fafbf8] text-[#748078] border border-[#e5e9e3]">
              Range: {formatSGD(pmc)} – {formatSGD(pme)}
            </span>
            {candidatePrice != null && candidatePrice > 0 && (
              <span className={`text-xs font-bold px-3 py-1 rounded-lg ${
                candidatePrice > pme ? "bg-[#fbefeb] text-[#c87970] border border-[#eed7d0]" : "bg-[#214f43] text-white"
              }`}>
                Pitch: {formatSGD(candidatePrice)}
              </span>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={curves}
              margin={{ top: 10, right: 20, left: -10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E9E3" opacity={0.8} />
              <XAxis
                dataKey="price"
                tickFormatter={(val) => `S$${val}`}
                tick={{ fontSize: 11, fill: "#748078", fontFamily: "DM Sans" }}
                stroke="#E5E9E3"
              />
              <YAxis
                tickFormatter={(val) => `${val}%`}
                domain={[0, 100]}
                tick={{ fontSize: 11, fill: "#748078", fontFamily: "DM Sans" }}
                stroke="#E5E9E3"
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="top"
                height={36}
                wrapperStyle={{ fontSize: "12px", paddingTop: "0px" }}
              />

              {/* Van Westendorp 4 Curves matching simsg-frontend colors */}
              <Line
                type="monotone"
                dataKey="too_cheap"
                name="Too Cheap"
                stroke="#8c79b4"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="cheap"
                name="Cheap / Bargain"
                stroke="#4d9076"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="expensive"
                name="Expensive"
                stroke="#d6ad69"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="too_expensive"
                name="Too Expensive"
                stroke="#c87970"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 4 }}
              />

              {/* OPP Marker */}
              <ReferenceLine
                x={opp}
                stroke="#4d9076"
                strokeDasharray="4 4"
                strokeWidth={1.8}
                label={{
                  value: `OPP: S$${opp.toFixed(2)}`,
                  position: "top",
                  fill: "#4d9076",
                  fontSize: 11,
                  fontWeight: "bold",
                }}
              />

              {/* Candidate Price Marker */}
              {candidatePrice != null && candidatePrice > 0 && (
                <ReferenceLine
                  x={candidatePrice}
                  stroke="#214f43"
                  strokeWidth={2}
                  label={{
                    value: `Pitch: S$${candidatePrice.toFixed(2)}`,
                    position: "insideTopLeft",
                    fill: "#214f43",
                    fontSize: 11,
                    fontWeight: "bold",
                  }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Analytical Pricing Insight Banner */}
        <div className="mt-4 rounded-xl bg-[#f8faf5] p-3.5 text-xs flex items-start gap-2.5 border border-[#dce7d2]">
          <Info className="h-4 w-4 text-[#4d9076] shrink-0 mt-0.5" />
          <div className="space-y-0.5 leading-relaxed">
            <span className="font-bold text-[#263c35]">Econometric Analysis: </span>
            <span className="text-[#748078]">
              The <strong>Optimal Price Point (OPP)</strong> is <strong>{formatSGD(opp)}</strong> where consumer resistance is minimized. The <strong>Indifference Price Point (IPP)</strong> sits at <strong>{formatSGD(ipp)}</strong>. The acceptable market range spans from <strong>{formatSGD(pmc)}</strong> (Point of Marginal Cheapness) to <strong>{formatSGD(pme)}</strong> (Point of Marginal Expensiveness).
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
