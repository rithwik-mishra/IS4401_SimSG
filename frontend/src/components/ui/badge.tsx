import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground shadow hover:bg-destructive/80",
        outline: "text-foreground",
        success:
          "border-transparent bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
        warning:
          "border-transparent bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300",
        risk:
          "border-transparent bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
        cohortSenior:
          "border-transparent bg-[#8C79B4]/15 text-[#5e4b85] font-medium",
        cohortWorking:
          "border-transparent bg-[#4D9076]/15 text-[#2a5b48] font-medium",
        cohortFamily:
          "border-transparent bg-[#D6AD69]/15 text-[#7a581e] font-medium",
        cohortYouth:
          "border-transparent bg-[#8FA861]/15 text-[#485d26] font-medium",
        cohortAffluent:
          "border-transparent bg-[#5B84B1]/15 text-[#2c4e73] font-medium",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };

