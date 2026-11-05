import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatSGD(amount: number): string {
  return `S$${amount.toFixed(2)}`;
}

/**
 * Household income from work as its census band. The census only gives the band, so the exact
 * drawn amount is never shown. Falls back to the amount for older results that have no band.
 *   full:    "S$7,000 to S$7,999" / "below S$1,000" / "S$35,000 or more"
 *   compact: "S$7k–8k"            / "<S$1k"          / "S$35k+"
 */
export function formatIncomeBand(
  persona: { income_bracket?: string | null; monthly_income_sgd: number },
  compact = false
): string {
  const band = persona.income_bracket;
  if (!band) return `S$${Math.round(persona.monthly_income_sgd).toLocaleString()}`;
  if (!compact) return band;
  // 7,000 -> "7k"; band tops end in 999 or 499: 7,999 -> "8k", 17,499 -> "17.5k"; 17,500 -> "17.5k"
  const k = (s: string) => {
    const n = Number(s.replace(/,/g, ""));
    const value = n % 500 === 0 ? n : n + 1;
    return `${Number((value / 1000).toFixed(1))}k`;
  };
  let m = band.match(/^below S\$([\d,]+)$/);
  if (m) return `<S$${k(m[1])}`;
  m = band.match(/^S\$([\d,]+) or more$/);
  if (m) return `S$${k(m[1])}+`;
  m = band.match(/^S\$([\d,]+) to S\$([\d,]+)$/);
  if (m) return `S$${k(m[1])}–${k(m[2])}`;
  return band;
}

export function formatPct(percent: number): string {
  return `${percent.toFixed(1)}%`;
}

