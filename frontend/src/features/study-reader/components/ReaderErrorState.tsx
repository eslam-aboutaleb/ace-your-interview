import { AlertTriangle } from "lucide-react";

interface ReaderErrorStateProps {
  message: string;
  onRetry: () => void;
}

/** Recoverable error surface for the reader (issue #7). */
export default function ReaderErrorState({ message, onRetry }: ReaderErrorStateProps) {
  return (
    <div
      role="alert"
      className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center"
    >
      <AlertTriangle className="w-8 h-8 text-udemy-warning" aria-hidden="true" />
      <p className="text-udemy-text font-medium">The reader could not load.</p>
      <p className="text-sm text-udemy-text-muted max-w-md">{message}</p>
      <button type="button" onClick={onRetry} className="btn-primary mt-2">
        Try again
      </button>
    </div>
  );
}
