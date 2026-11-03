import React from "react";
import { cn } from "@/lib/utils";
import { Coffee, Utensils, Beer, Sparkles, MapPin } from "lucide-react";
import { PresetQuery } from "@/data/presets";

export interface SuggestionsProps {
  suggestions: PresetQuery[];
  onSelect: (preset: PresetQuery) => void;
  className?: string;
}

export function Suggestions({
  suggestions,
  onSelect,
  className,
}: SuggestionsProps) {
  const getIcon = (category: string) => {
    if (category.includes("Beverage") || category.includes("Cafe")) {
      return <Coffee className="h-3.5 w-3.5 text-[#bda567]" />;
    }
    if (category.includes("Bar") || category.includes("Alcohol")) {
      return <Beer className="h-3.5 w-3.5 text-[#d6ad69]" />;
    }
    if (category.includes("Bakery") || category.includes("Dining")) {
      return <Utensils className="h-3.5 w-3.5 text-[#4d9076]" />;
    }
    return <Sparkles className="h-3.5 w-3.5 text-[#6b9a90]" />;
  };

  return (
    <div className={cn("space-y-2.5", className)}>
      <div className="flex items-center gap-1.5 text-[11px] font-bold text-[#748078] uppercase tracking-[0.1em]">
        <Sparkles className="h-3.5 w-3.5 text-[#214f43]" />
        <span>Try Singapore Market Scenarios</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onSelect(preset)}
            className="group flex items-center gap-2.5 rounded-xl border border-[#e5e9e3] bg-white px-3.5 py-2 text-xs font-medium text-[#263c35] transition-all hover:border-[#214f43]/40 hover:bg-[#fafbf8] hover:shadow-xs cursor-pointer active:scale-[0.98]"
          >
            <div className="rounded-lg bg-[#f5f6f2] p-1.5 group-hover:bg-[#e8efdf]/60 transition-colors">
              {getIcon(preset.category)}
            </div>
            <div className="text-left">
              <span className="font-semibold text-[#263c35] group-hover:text-[#214f43] block">
                {preset.title}
              </span>
              <span className="text-[#748078] text-[11px] flex items-center gap-1 mt-0.5">
                <MapPin className="h-3 w-3 text-[#748078]" />
                {preset.town} • S${preset.price.toFixed(2)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
