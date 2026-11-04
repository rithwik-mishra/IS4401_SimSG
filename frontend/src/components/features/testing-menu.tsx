import React from "react";
import { FlaskConical, Copy, Check, X } from "lucide-react";

// Seeds are whole numbers from 0 to 2,147,483,647 (matches the backend's limit).
export const MAX_SEED = 2_147_483_647;

export interface TestingMenuProps {
  seedInput: string;
  onSeedInputChange: (value: string) => void;
  lastSeed?: number | null;
  disabled?: boolean;
}

/**
 * Testing-only controls, tucked behind a button in the top bar.
 * The seed fixes which personas are drawn, so a run can be repeated.
 */
export function TestingMenu({ seedInput, onSeedInputChange, lastSeed, disabled }: TestingMenuProps) {
  const [open, setOpen] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

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
        className="relative flex h-9 items-center gap-1.5 rounded-md border border-white/20 bg-white/10 px-2.5 text-xs text-white hover:bg-white/20 transition-all cursor-pointer"
      >
        <FlaskConical className="h-4 w-4 text-emerald-200" />
        <span className="hidden sm:inline">Testing</span>
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
