import * as React from "react";
import { cn } from "../../lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "secondary" | "success" | "warning" | "muted";
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(({ className, variant = "default", ...props }, ref) => {
  const colorStyles = {
    default: "bg-white/5 dark:text-slate-100 text-slate-900",
    secondary: "dark:bg-slate-800 dark:text-slate-200 bg-slate-200 text-slate-700",
    success: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
    warning: "bg-amber-500/15 text-amber-600 dark:text-amber-300",
    muted: "dark:bg-slate-800 dark:text-slate-300 bg-slate-100 text-slate-600",
  };

  return (
    <span
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium",
        colorStyles[variant],
        className
      )}
      {...props}
    />
  );
});
Badge.displayName = "Badge";

export { Badge };
