/**
 * Shared async data-fetching hook.
 *
 * Centralizes the `{data, error, loading, retry}` pattern
 * that ~15 pages previously re-implemented with ad-hoc
 * useEffect + useState blocks. Features:
 *
 * - abort-on-unmount: each run creates an AbortController
 *   and aborts it when the component unmounts or the
 *   `deps` change; fetchers that accept a signal cancel
 *   the in-flight request, and state updates from aborted
 *   runs are discarded.
 * - shared error-message extraction via `extractApiErrorMessage`
 *   (or a fixed message with `fixedErrorMessage`).
 * - optional `deps` refetch, mirroring useEffect semantics.
 *
 * Fetchers that do not support AbortSignal simply ignore
 * the signal argument.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { extractApiErrorMessage } from "@/utils/apiError";

export interface UseApiOptions<T> {
  /** Refetch whenever any dependency changes (useEffect-style). */
  deps?: readonly unknown[];
  /** Fetch immediately on mount. Default: true. */
  immediate?: boolean;
  /** Message shown when the fetcher rejects. */
  errorMessage?: string;
  /**
   * When true, show `errorMessage` verbatim instead of
   * extracting the server detail from the error.
   */
  fixedErrorMessage?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (message: string) => void;
}

export interface UseApiResult<T> {
  data: T | null;
  error: string;
  loading: boolean;
  /** Re-run the fetcher. */
  retry: () => void;
  /** Overwrite the cached data (e.g. after a local mutation). */
  setData: (
    updater: T | null | ((prev: T | null) => T | null),
  ) => void;
}

export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  options: UseApiOptions<T> = {},
): UseApiResult<T> {
  const {
    deps = [],
    immediate = true,
    errorMessage = "Request failed.",
    fixedErrorMessage = false,
    onSuccess,
    onError,
  } = options;

  const [data, setDataState] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(immediate);

  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const onSuccessRef = useRef(onSuccess);
  onSuccessRef.current = onSuccess;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const errorMessageRef = useRef(errorMessage);
  errorMessageRef.current = errorMessage;
  const fixedErrorMessageRef = useRef(fixedErrorMessage);
  fixedErrorMessageRef.current = fixedErrorMessage;
  const immediateRef = useRef(immediate);
  immediateRef.current = immediate;

  const run = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError("");
    try {
      const result = await fetcherRef.current(controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return;
      setDataState(result);
      onSuccessRef.current?.(result);
    } catch (err) {
      if (!mountedRef.current || controller.signal.aborted) return;
      const message = fixedErrorMessageRef.current
        ? errorMessageRef.current
        : extractApiErrorMessage(err, errorMessageRef.current);
      setError(message);
      onErrorRef.current?.(message);
    } finally {
      if (mountedRef.current && !controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (immediateRef.current) {
      void run();
    }
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
    // `deps` is caller-controlled (useEffect-style refetch list).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps]);

  const setData = useCallback(
    (updater: T | null | ((prev: T | null) => T | null)) => {
      setDataState((prev) =>
        typeof updater === "function"
          ? (updater as (prev: T | null) => T | null)(prev)
          : updater,
      );
    },
    [],
  );

  return { data, error, loading, retry: run, setData };
}
