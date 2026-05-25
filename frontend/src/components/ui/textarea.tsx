import * as React from "react";
import { cn } from "../../lib/utils";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "min-h-[180px] w-full rounded-3xl border dark:border-white/10 border-slate-300 dark:bg-surface-800 bg-white/90 px-4 py-3 dark:text-slate-100 text-slate-900 outline-none transition focus:border-accent-400 focus:ring-2 focus:ring-accent-500/20",
      className
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";

export { Textarea };
