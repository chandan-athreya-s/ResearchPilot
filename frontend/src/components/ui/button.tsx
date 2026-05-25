import * as React from "react";
import { cn } from "../../lib/utils";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "ghost";
  size?: "default" | "sm";
}

const variantStyles: Record<NonNullable<ButtonProps["variant"]>, string> = {
  default: "bg-accent-500 text-white hover:bg-accent-400",
  secondary: "bg-surface-800 text-slate-200 hover:bg-surface-700",
  ghost: "bg-transparent text-slate-200 hover:bg-white/5",
};

const sizeStyles: Record<NonNullable<ButtonProps["size"]>, string> = {
  default: "h-11 px-5 py-2.5 text-sm",
  sm: "h-9 px-4 text-sm",
};

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center rounded-full font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:pointer-events-none disabled:opacity-50",
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";

export { Button };
