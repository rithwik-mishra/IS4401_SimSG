import React from "react";
import { FlaskConical, Copy, Check, X } from "lucide-react";
import { getAIStatus, type AIStatus } from "../../services/api";
import type { AIUsage } from "../../types/api";

/** One row per AI step of the last run: green = AI answered, amber = rule-based backup (fully or partly). */
function LastRunSteps({ usage }: { usage?: AIUsage | null }) {
  const rows: { label: string; usedAI: boolean | null; text: string }[] = usage
    ? [
        { label: "1. Pitch reader", usedAI: usage.pitch_reader === "ai", text: usage.pitch_reader === "ai" ? "AI" : "Backup" },
        {
          label: "4. Persona answers",
          usedAI: usage.persona_backup === 0,
          text:
            usage.persona_backup === 0
              ? `AI ${usage.persona_ai}/${usage.persona_ai}`
              : usage.persona_ai === 0
                ? `Backup ${usage.persona_backup}/${usage.persona_backup}`
                : `AI ${usage.persona_ai} · Backup ${usage.persona_backup}`,
        },
        { label: "6. Report writer", usedAI: usage.report_writer === "ai", text: usage.report_writer === "ai" ? "AI" : "Backup" },
      ]
    : [
        { label: "1. Pitch reader", usedAI: null, text: "—" },
        { label: "4. Persona answers", usedAI: null, text: "—" },
        { label: "6. Report writer", usedAI: null, text: "—" },
      ];

  return (
    <div className="rounded-lg bg-[#f8faf5] border border-[#e5e9e3] px-3 py-2 text-xs space-y-1" data-testid="last-run-steps">
      <div className="text-[#748078]">Last run</div>
      {rows.map((row) => (
        <div key={row.label} className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 text-[#263c35]">
            <span
              className={`h-2 w-2 shrink-0 rounded-full ${
                row.usedAI === null ? "bg-[#c5cbc4]" : row.usedAI ? "bg-emerald-500" : "bg-amber-400"
              }`}
              aria-hidden="true"
            />
            {row.label}
          </span>
          <span className="font-mono font-semibold text-[#263c35]">{row.text}</span>
        </div>
      ))}
    </div>
  );
}

type AIStatusState = { kind: "loading" } | { kind: "ok"; status: AIStatus } | { kind: "error" };

/** Shows which AI model the backend would use, or blank when it would use the backup logic. */
function AIModelIndicator({ state }: { state: AIStatusState }) {
  let dot = "bg-[#c5cbc4]";
  let model = "—";
  let note = "Checking…";

  if (state.kind === "error") {
    note = "Backend not reachable";
  } else if (state.kind === "ok" && state.status.persona_engine === "llm") {
    dot = "bg-emerald-500";
    model = state.status.llm_model ?? "—";
    note = `AI (${state.status.llm_provider})`;
  } else if (state.kind === "ok") {
    dot = "bg-amber-400";
    note = "Backup logic (no API key)";
  }

  return (
    <div className="flex items-center justify-between gap-2 rounded-lg bg-[#f8faf5] border border-[#e5e9e3] px-3 py-2 text-xs">
      <span className="flex items-center gap-1.5 text-[#748078]">
        <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} aria-hidden="true" />
        AI model:{" "}
        <span className="font-mono font-semibold text-[#263c35]" data-testid="ai-model">{model}</span>
      </span>
      <span className="text-[11px] text-[#748078] text-right">{note}</span>
    </div>
  );
}

// Seeds are whole numbers from 0 to 2,147,483,647 (matches the backend's limit).
export const MAX_SEED = 2_147_483_647;

export interface TestingMenuProps {
  seedInput: string;
  onSeedInputChange: (value: string) => void;
  lastSeed?: number | null;
  lastAIUsage?: AIUsage | null;
  disabled?: boolean;
}

/**
 * Testing-only controls, tucked behind a button in the top bar.
 * The seed fixes which personas are drawn, so a run can be repeated.
 */
export function TestingMenu({ seedInput, onSeedInputChange, lastSeed, lastAIUsage, disabled }: TestingMenuProps) {
  const [open, setOpen] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [aiStatus, setAIStatus] = React.useState<AIStatusState>({ kind: "loading" });

  // Ask the backend each time the panel opens, so a restart with a new key shows up.
  React.useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setAIStatus({ kind: "loading" });
    getAIStatus(controller.signal)
      .then((status) => setAIStatus({ kind: "ok", status }))
      .catch((err) => {
        if (err?.name !== "AbortError") setAIStatus({ kind: "error" });
      });
    return () => controller.abort();
  }, [open]);

  // Close when clicking outside the panel or pressing Escape.
  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  // Keep only digits, capped at the backend's maximum seed.
  const handleChange = (raw: string) => {
    const digits = raw.replace(/\D/g, "").slice(0, 10);
    if (digits !== "" && Number(digits) > MAX_SEED) return;
    onSeedInputChange(digits);
  };

  const handleCopy = async () => {
    if (lastSeed == null) return;
    try {
      await navigator.clipboard.writeText(String(lastSeed));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be blocked; the seed is still visible to copy by hand.
    }
  };

  const seedIsSet = seedInput !== "";

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Testing tools"
        aria-label="Testing tools"
        className="relative flex h-9 w-9 items-center justify-center rounded-md text-white/70 hover:bg-white/10 hover:text-white transition-all cursor-pointer"
      >
        <FlaskConical className="h-4 w-4" />
        {/* Dot reminds you a fixed seed is active, so every run redraws the same personas */}
        {seedIsSet && (
          <span className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-amber-400 ring-2 ring-[#214f43]" aria-label="Seed is set" />
        )}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Testing tools"
          className="absolute right-0 top-11 z-50 w-[min(18rem,calc(100vw-2rem))] rounded-xl border border-[#e5e9e3] bg-white p-4 text-[#263c35] shadow-lg space-y-3"
        >
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold tracking-[0.12em] text-[#748078] uppercase">Testing</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close testing tools"
              className="rounded p-1 text-[#748078] hover:bg-[#e8efdf] cursor-pointer"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <AIModelIndicator state={aiStatus} />
          <LastRunSteps usage={lastAIUsage} />

          <label className="block space-y-1.5">
            <span className="text-xs font-semibold">Seed</span>
            <div className="flex items-center gap-2">
              <input
                type="text"
                inputMode="numeric"
                value={seedInput}
                onChange={(e) => handleChange(e.target.value)}
                disabled={disabled}
                placeholder="Empty = random"
                className="h-9 min-w-0 flex-1 rounded-md border border-[#e0e6dd] bg-white px-3 text-sm font-semibold outline-none placeholder:font-normal placeholder:text-[#99a39a] focus:border-[#214f43] disabled:opacity-60"
              />
              {seedIsSet && (
                <button
                  type="button"
                  onClick={() => onSeedInputChange("")}
                  disabled={disabled}
                  className="h-9 rounded-md px-2.5 text-xs text-[#748078] hover:bg-[#e8efdf] cursor-pointer disabled:opacity-60"
                >
                  Clear
                </button>
              )}
            </div>
            <span className="block text-[11px] leading-snug text-[#748078]">
              The same seed redraws the same personas. With no API key the whole result repeats.
            </span>
          </label>

          <div className="flex items-center justify-between gap-2 rounded-lg bg-[#f8faf5] border border-[#e5e9e3] px-3 py-2 text-xs">
            <span className="text-[#748078]">
              Last run seed:{" "}
              <span className="font-mono font-semibold text-[#263c35]">{lastSeed ?? "—"}</span>
            </span>
            {lastSeed != null && (
              <span className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={handleCopy}
                  title="Copy seed"
                  aria-label="Copy seed"
                  className="rounded p-1 hover:bg-[#e8efdf] cursor-pointer"
                >
                  {copied ? <Check className="h-3.5 w-3.5 text-[#214f43]" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
                <button
                  type="button"
                  onClick={() => onSeedInputChange(String(lastSeed))}
                  disabled={disabled}
                  title="Put this seed in the seed box"
                  className="rounded px-1.5 py-0.5 font-medium text-[#214f43] hover:bg-[#e8efdf] cursor-pointer disabled:opacity-60"
                >
                  Use
                </button>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
