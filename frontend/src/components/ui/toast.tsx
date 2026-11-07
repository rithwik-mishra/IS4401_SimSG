import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { CheckCircle2, AlertCircle, X, Sparkles } from "lucide-react";

export type ToastType = "success" | "error" | "info";

export interface ToastProps {
  isOpen: boolean;
  title: string;
  description?: string;
  type?: ToastType;
  onClose: () => void;
  duration?: number;
}

export function Toast({
  isOpen,
  title,
  description,
  type = "info",
  onClose,
  duration = 5000,
}: ToastProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const remainingRef = useRef<number>(duration);
  const startTimeRef = useRef<number>(Date.now());

  const startTimer = () => {
    if (duration <= 0) return;
    startTimeRef.current = Date.now();
    timerRef.current = setTimeout(() => {
      onClose();
    }, remainingRef.current);
  };

  const pauseTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
      const elapsed = Date.now() - startTimeRef.current;
      remainingRef.current = Math.max(0, remainingRef.current - elapsed);
    }
  };

  useEffect(() => {
    if (isOpen) {
      remainingRef.current = duration;
      startTimer();
    } else {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isOpen, duration]);

  const typeConfig = {
    success: {
      icon: CheckCircle2,
      badgeColor: "bg-[#edf4e8] text-[#214f43] border-[#dce7d2]",
      borderColor: "border-[#dce7d2]",
      accentGlow: "shadow-[0_10px_30px_-5px_rgba(33,79,67,0.15)]",
    },
    error: {
      icon: AlertCircle,
      badgeColor: "bg-[#fbefeb] text-[#c87970] border-[#eed7d0]",
      borderColor: "border-[#eed7d0]",
      accentGlow: "shadow-[0_10px_30px_-5px_rgba(200,121,112,0.15)]",
    },
    info: {
      icon: Sparkles,
      badgeColor: "bg-[#edf3f8] text-[#2b5c8f] border-[#d4e2f0]",
      borderColor: "border-[#d4e2f0]",
      accentGlow: "shadow-[0_10px_30px_-5px_rgba(43,92,143,0.15)]",
    },
  }[type];

  const IconComponent = typeConfig.icon;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.94 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 16, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 420, damping: 32 }}
          role={type === "error" ? "alert" : "status"}
          aria-live={type === "error" ? "assertive" : "polite"}
          onMouseEnter={pauseTimer}
          onMouseLeave={startTimer}
          className={`fixed bottom-5 right-5 sm:bottom-6 sm:right-6 z-50 w-[calc(100vw-2.5rem)] max-w-md rounded-2xl border bg-white/95 backdrop-blur-md p-4 shadow-xl ${typeConfig.borderColor} ${typeConfig.accentGlow} transition-all`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`h-9 w-9 rounded-xl flex items-center justify-center shrink-0 border ${typeConfig.badgeColor}`}
            >
              <IconComponent className="h-5 w-5" />
            </div>

            <div className="flex-1 pt-0.5 space-y-1 pr-1">
              <div className="flex items-center gap-1.5">
                <span className="font-heading font-bold text-sm text-[#263c35]">
                  {title}
                </span>
                {type === "success" && (
                  <Sparkles className="h-3.5 w-3.5 text-[#4d9076]" />
                )}
              </div>
              {description && (
                <p className="text-xs text-[#586860] leading-relaxed">
                  {description}
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={onClose}
              aria-label="Close notification"
              className="p-1 rounded-lg text-[#748078] hover:text-[#263c35] hover:bg-black/5 transition-colors focus:outline-none focus:ring-2 focus:ring-[#214f43]/30"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
