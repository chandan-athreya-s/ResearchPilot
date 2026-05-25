import * as React from "react";
import { cn } from "../../lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "w-full rounded-2xl border border-white/10 bg-surface-800 px-4 py-3 text-slate-100 outline-none transition focus:border-accent-400 focus:ring-2 focus:ring-accent-500/20",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";

export { Input };
