import React from "react";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

export interface TopNavigationProps {
  onReset: () => void;
  hasResults: boolean;
  // Extra controls shown at the far right (e.g. the testing menu)
  rightSlot?: React.ReactNode;
}

export function TopNavigation({
  onReset,
  hasResults,
  rightSlot,
}: TopNavigationProps) {
  return (
    <header className="sticky top-0 z-40 w-full bg-[#214f43] text-white border-b border-[#183d34] shadow-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand Identity with Demographic Scanning Radar Vector Mark */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl overflow-hidden shadow-md ring-1 ring-emerald-500/20 hover:scale-105 transition-transform cursor-pointer">
            <img
              src="/mark.svg"
              alt="SimSG"
              className="h-full w-full object-contain"
            />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold tracking-tight text-white text-xl font-heading">
                SimSG<span className="text-[#81a563]">.</span>
              </span>
            </div>
            <p className="text-[11px] text-emerald-100/80 hidden sm:block">
              Singapore Consumer Demographic & Economic Simulation
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          {hasResults && (
            <Button
              variant="outline"
              size="sm"
              onClick={onReset}
              className="text-xs bg-white/10 text-white border-white/20 hover:bg-white/20 hover:text-white transition-all flex items-center gap-1.5 shadow-xs"
            >
              <RefreshCw className="h-3.5 w-3.5 text-emerald-200" />
              <span>New Simulation</span>
            </Button>
          )}
          {rightSlot}
        </div>
      </div>
    </header>
  );
}
