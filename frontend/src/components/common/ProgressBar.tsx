import { motion } from "framer-motion";
import { progressSpring } from "@/utils/animations";

interface Props {
  percent: number;
  className?: string;
  showLabel?: boolean;
}

export default function ProgressBar({
  percent,
  className = "",
  showLabel = false,
}: Props) {
  return (
    <div className={`progress-bar ${className}`}>
      <motion.div
        className="progress-bar-fill"
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        transition={progressSpring}
      />
      {showLabel && (
        <span className="text-xs text-udemy-text-muted ml-2">
          {Math.round(percent)}%
        </span>
      )}
    </div>
  );
}
