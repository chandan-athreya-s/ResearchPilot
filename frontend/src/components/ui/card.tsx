import * as React from "react";
import { cn } from "../../lib/utils";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

const Card = React.forwardRef<HTMLDivElement, CardProps>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-3xl border dark:border-white/10 border-slate-200/60 dark:bg-surface-900/95 bg-white/95 p-5 shadow-soft backdrop-blur-xl",
      className
    )}
    {...props}
  />
));
Card.displayName = "Card";

export { Card };
