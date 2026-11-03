import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatSGD(amount: number): string {
  return `S$${amount.toFixed(2)}`;
}

export function formatPct(percent: number): string {
  return `${percent.toFixed(1)}%`;
}

