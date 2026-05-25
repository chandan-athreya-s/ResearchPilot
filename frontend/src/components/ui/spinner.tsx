import { motion } from "framer-motion";

export function Spinner() {
  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ repeat: Infinity, duration: 1.1, ease: "linear" }}
      className="inline-flex h-10 w-10 items-center justify-center rounded-full border-2 border-accent-400/20 border-t-accent-400"
    />
  );
}
