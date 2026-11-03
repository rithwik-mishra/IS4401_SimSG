import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center cursor-pointer py-2",
      className
    )}
    {...props}
  >
    {/* Explicitly visible high-contrast track */}
    <SliderPrimitive.Track className="relative h-2.5 w-full grow overflow-hidden rounded-full bg-[#e0e6dd] border border-[#d2ded0]">
      {/* Explicitly visible solid forest green range */}
      <SliderPrimitive.Range className="absolute h-full bg-[#214f43]" />
    </SliderPrimitive.Track>
    {/* Tactile visible thumb */}
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-[#214f43] bg-white shadow-md transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#214f43] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 cursor-grab active:cursor-grabbing active:scale-95" />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
