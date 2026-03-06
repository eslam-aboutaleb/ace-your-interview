import { motion } from "framer-motion";
import { pulseVariants } from "@/utils/animations";

interface Props {
  count?: number;
}

export default function SkeletonCards({ count = 6 }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className="udemy-card p-5"
          variants={pulseVariants}
          animate="pulse"
        >
          <div className="skeleton h-4 w-3/4 mb-3" />
          <div className="skeleton h-3 w-full mb-2" />
          <div className="skeleton h-3 w-5/6 mb-4" />
          <div className="skeleton h-2 w-full mb-2" />
          <div className="flex justify-between">
            <div className="skeleton h-3 w-20" />
            <div className="skeleton h-3 w-16" />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
