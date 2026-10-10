/**
 * Route-level error boundary.
 *
 * Catches render errors in route elements so a single
 * broken page shows a recoverable fallback instead of
 * a blank screen. Wrap route elements in `App.tsx`.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Custom fallback; defaults to the built-in card UI. */
  fallback?: (props: {
    error: Error;
    reset: () => void;
  }) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error, info.componentStack);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback({
          error: this.state.error,
          reset: this.reset,
        });
      }
      return (
        <div className="min-h-[40vh] flex items-center justify-center p-6">
          <div className="udemy-card p-8 max-w-md w-full text-center">
            <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto mb-3" />
            <h2 className="text-lg font-bold mb-2">
              Something went wrong
            </h2>
            <p className="text-sm text-udemy-text-muted mb-4">
              {this.state.error.message ||
                "An unexpected error occurred."}
            </p>
            <button
              onClick={this.reset}
              className="btn-primary inline-flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
