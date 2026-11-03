import * as React from "react";
import { cn } from "@/lib/utils";
import { Sparkles, Loader2, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
  children?: React.ReactNode;
}

export function PromptInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  placeholder = "Describe your concept, target location, price point, or menu...",
  className,
  children,
}: PromptInputProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        240
      )}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div
      className={cn(
        "relative rounded-2xl border border-[#e0e6dd] bg-white shadow-md transition-all focus-within:border-[#214f43] focus-within:ring-2 focus-within:ring-[#214f43]/15",
        className
      )}
    >
      <div className="p-4 sm:p-5">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={2}
          disabled={isLoading}
          className="w-full resize-none bg-transparent text-sm sm:text-base leading-relaxed text-[#263c35] outline-none placeholder:text-[#99a39a] disabled:opacity-60"
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#f0f3ee] bg-[#fafbf8] px-4 py-3 rounded-b-2xl">
        <div className="flex flex-wrap items-center gap-2.5">{children}</div>

        <Button
          onClick={onSubmit}
          disabled={!value.trim() || isLoading}
          size="sm"
          className="rounded-xl px-5 py-2.5 font-semibold bg-[#214f43] hover:bg-[#183d34] text-white shadow-sm flex items-center gap-2 transition-all cursor-pointer active:scale-[0.98]"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-emerald-200" />
              <span>Simulating...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4 text-[#81a563]" />
              <span>Simulate</span>
              <ArrowUpRight className="h-4 w-4 opacity-80" />
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
