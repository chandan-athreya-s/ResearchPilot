import * as React from "react";
import { cn } from "../../lib/utils";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "min-h-[180px] w-full rounded-3xl border border-white/10 bg-surface-800 px-4 py-3 text-slate-100 outline-none transition focus:border-accent-400 focus:ring-2 focus:ring-accent-500/20",
      className
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";

export { Textarea };
