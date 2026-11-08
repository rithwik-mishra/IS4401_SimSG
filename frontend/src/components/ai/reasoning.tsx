import * as React from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, CheckCircle2, Loader2, Sparkles, BrainCircuit, AlertCircle } from "lucide-react";

export interface PipelineStep {
  id: string;
  title: string;
  description: string;
  status: "pending" | "running" | "completed" | "error";
  errorMessage?: string;
}

export interface ReasoningProps {
  steps: PipelineStep[];
  currentStepIndex: number;
  activeStatusHint?: string;
  isOpen?: boolean;
  onToggle?: () => void;
  className?: string;
}

export function Reasoning({
  steps,
  currentStepIndex,
  activeStatusHint,
  isOpen = true,
  onToggle,
  className,
}: ReasoningProps) {
  const [open, setOpen] = React.useState(isOpen);

  const toggle = () => {
    if (onToggle) onToggle();
    else setOpen(!open);
  };

  const completedCount = steps.filter((s) => s.status === "completed").length;
  const isAllCompleted = completedCount === steps.length;
  const hasError = steps.some((s) => s.status === "error");
  const runningStepIndex = steps.findIndex((s) => s.status === "running");
  const isRunning = runningStepIndex !== -1;

  // Active step index: fallback to currentStepIndex or runningStepIndex
  const activeIdx = runningStepIndex !== -1 ? runningStepIndex : currentStepIndex;

  // Calculate high-fidelity progress percentage
  // When a step is running, allocate partial progress so the bar continuously feels alive
  const rawPercent = isAllCompleted
    ? 100
    : Math.round(((completedCount + (isRunning ? 0.6 : 0)) / steps.length) * 100);
  const progressPercent = Math.min(rawPercent, isAllCompleted ? 100 : 96);

  return (
    <div
      className={cn(
        "rounded-2xl border border-[#d5ded3] bg-white shadow-sm overflow-hidden transition-all",
        className
      )}
    >
      {/* Top Header Card */}
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="w-full flex items-center justify-between p-4 bg-[#f8faf7] border-b border-[#e5e9e3] hover:bg-[#f2f6f0] transition-colors cursor-pointer text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#214f43]"
      >
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border",
              hasError
                ? "bg-[#fbefeb] border-[#eed7d0]"
                : "bg-[#e8efdf] border-[#d2ded0]"
            )}
          >
            {hasError ? (
              <AlertCircle className="h-4.5 w-4.5 text-[#c87970]" />
            ) : (
              <BrainCircuit
                className={cn(
                  "h-4.5 w-4.5 text-[#214f43]",
                  !isAllCompleted ? "animate-pulse" : ""
                )}
              />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-heading font-bold text-sm text-[#263c35]">
                Multi-Agent Simulation Pipeline
              </span>
              <span
                className={cn(
                  "rounded-full px-2.5 py-0.5 text-[11px] font-bold border transition-colors",
                  hasError
                    ? "bg-[#fbefeb] text-[#8b423b] border-[#eed7d0]"
                    : isAllCompleted
                    ? "bg-[#edf4e8] text-[#2d7a5b] border-[#cde0cf]"
                    : "bg-[#e8efdf] text-[#1b4337] border-[#cedbc3]"
                )}
              >
                {hasError
                  ? "Pipeline Halted"
                  : isAllCompleted
                  ? "All 6 Stages Completed"
                  : activeIdx >= 0
                  ? `Stage ${activeIdx + 1} of ${steps.length} in Progress`
                  : `${completedCount}/${steps.length} Stages`}
              </span>
            </div>
            {activeStatusHint && !isAllCompleted && !hasError && (
              <p className="text-[11px] font-medium text-[#2d4b3e] mt-1 flex items-center gap-1.5 animate-pulse">
                <span className="w-1.5 h-1.5 rounded-full bg-[#214f43] inline-block shrink-0" />
                {activeStatusHint}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2.5 shrink-0 ml-3">
          <span className="text-xs font-bold text-[#214f43] tabular-nums">
            {progressPercent}% Complete
          </span>
          <ChevronDown
            className={cn(
              "h-4 w-4 text-[#263c35] transition-transform duration-200",
              open ? "rotate-180" : ""
            )}
          />
        </div>
      </button>

      {/* Progress Track */}
      <div className="px-4 py-2 bg-[#fbfcfb] border-b border-[#edf1eb]">
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-[#e3eae0]">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500 ease-out",
              hasError ? "bg-[#c87970]" : "bg-[#214f43]"
            )}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Step by Step Breakdown */}
      {open && (
        <div className="divide-y divide-[#edf1eb] bg-white">
          {steps.map((step, idx) => {
            const isStepCompleted = step.status === "completed";
            const isStepRunning = step.status === "running";
            const isStepError = step.status === "error";

            return (
              <div
                key={step.id}
                className={cn(
                  "flex items-start gap-3.5 p-3.5 sm:px-4 transition-all",
                  isStepError
                    ? "bg-[#fff8f7] border-l-4 border-l-[#c87970]"
                    : isStepRunning
                    ? "bg-[#f2f7f3] border-l-4 border-l-[#214f43]"
                    : isStepCompleted
                    ? "bg-white hover:bg-[#fafbfa] border-l-4 border-l-transparent"
                    : "bg-[#fcfdfc]/60 border-l-4 border-l-transparent"
                )}
              >
                {/* Step Status Icon */}
                <div className="mt-0.5 shrink-0">
                  {isStepError ? (
                    <div className="w-5 h-5 rounded-full bg-[#fbefeb] text-[#c87970] border border-[#eed7d0] flex items-center justify-center">
                      <AlertCircle className="h-3.5 w-3.5 text-[#c87970]" />
                    </div>
                  ) : isStepCompleted ? (
                    <div className="w-5 h-5 rounded-full bg-[#edf4e8] text-[#2d7a5b] border border-[#cde0cf] flex items-center justify-center">
                      <CheckCircle2 className="h-3.5 w-3.5 text-[#2d7a5b]" />
                    </div>
                  ) : isStepRunning ? (
                    <div className="w-5 h-5 rounded-full bg-[#214f43] text-white flex items-center justify-center shadow-xs">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 rounded-full border border-[#cbd5c8] bg-[#f4f7f3] text-[#4f6357] text-[11px] font-bold flex items-center justify-center">
                      {idx + 1}
                    </div>
                  )}
                </div>

                {/* Step Title & Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className={cn(
                        "text-sm tracking-tight",
                        isStepError
                          ? "font-bold text-[#8b423b]"
                          : isStepRunning
                          ? "font-bold text-[#143d32]"
                          : isStepCompleted
                          ? "font-semibold text-[#263c35]"
                          : "font-medium text-[#4a5c53]"
                      )}
                    >
                      {step.title}
                    </span>
                    {isStepError ? (
                      <span className="text-[10px] font-bold text-[#c87970] bg-[#fbefeb] border border-[#eed7d0] px-2 py-0.5 rounded-full shrink-0">
                        Halted
                      </span>
                    ) : isStepRunning ? (
                      <span className="text-[10px] uppercase font-bold tracking-wider bg-[#214f43] text-white px-2 py-0.5 rounded-full flex items-center gap-1 shrink-0 shadow-xs">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-300 animate-ping inline-block" />
                        Active
                      </span>
                    ) : isStepCompleted ? (
                      <span className="text-[10px] font-semibold text-[#2d7a5b] bg-[#edf4e8] border border-[#d2ded0] px-2 py-0.5 rounded-full shrink-0">
                        Done
                      </span>
                    ) : (
                      <span className="text-[10px] font-medium text-[#65766c] bg-[#f4f7f3] border border-[#d8e0d5] px-2 py-0.5 rounded-full shrink-0">
                        Queued
                      </span>
                    )}
                  </div>

                  <p
                    className={cn(
                      "text-xs mt-1 leading-relaxed",
                      isStepError
                        ? "text-[#8b423b]/90 font-normal"
                        : isStepRunning
                        ? "text-[#263c35] font-medium"
                        : isStepCompleted
                        ? "text-[#4a5e55]"
                        : "text-[#5e7065]"
                    )}
                  >
                    {step.description}
                  </p>

                  {/* Sub-hint when error or active */}
                  {isStepError && step.errorMessage ? (
                    <div className="mt-2.5 flex items-center gap-1.5 text-[11px] font-semibold text-[#8b423b] bg-[#fff0ed] border border-[#eed7d0] rounded-md px-2.5 py-1 w-fit shadow-2xs">
                      <AlertCircle className="h-3 w-3 text-[#c87970] shrink-0" />
                      <span>{step.errorMessage}</span>
                    </div>
                  ) : isStepRunning && activeStatusHint ? (
                    <div className="mt-2.5 flex items-center gap-1.5 text-[11px] font-semibold text-[#164d3d] bg-white border border-[#cde0cf] rounded-md px-2.5 py-1 w-fit shadow-2xs">
                      <Sparkles className="h-3 w-3 text-[#214f43] shrink-0 animate-spin" />
                      <span>{activeStatusHint}</span>
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
