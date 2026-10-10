/**
 * Shared stream-generation lifecycle with fallback recovery.
 *
 * The ndjson stream endpoints (`generateQuizV2Stream`,
 * `generateQuestionsV2Stream`, `createInterviewSessionStream`,
 * `submitInterviewAnswerStream`,
 * `generateNextInterviewQuestionStream`) reject with typed
 * `*StreamError` objects carrying a `fallbackEligible` flag.
 * When the flag is set, the caller may transparently retry
 * the equivalent non-streaming endpoint; when it is not,
 * the error must surface to the user. This hook centralizes
 * that recovery so pages no longer copy-paste the same
 * try/catch block.
 */

import { useCallback, useRef, useState } from "react";

export interface StreamGenerationResult<T> {
  data: T | null;
  usedFallback: boolean;
  error: unknown;
  /** True when the run was aborted (unmount or superseding run). */
  aborted: boolean;
}

export interface StreamGenerationOptions<TStream, TFallback> {
  /** Start the streaming request. Receives the run's AbortSignal. */
  stream: (signal: AbortSignal) => Promise<TStream>;
  /**
   * Non-streaming fallback invoked when the stream rejects
   * with a fallback-eligible error. Omit to keep partial
   * streamed results instead (e.g. quiz generation).
   */
  fallback?: () => Promise<TFallback>;
  /** Predicate deciding whether a stream error may fall back. */
  isFallbackEligible: (error: unknown) => boolean;
}

export function useStreamGeneration<
  TStream,
  TFallback = TStream,
>() {
  const [generating, setGenerating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (
      options: StreamGenerationOptions<TStream, TFallback>,
    ): Promise<StreamGenerationResult<TStream | TFallback>> => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setGenerating(true);
      try {
        const data = await options.stream(controller.signal);
        return {
          data,
          usedFallback: false,
          error: null,
          aborted: false,
        };
      } catch (streamErr) {
        if (controller.signal.aborted) {
          return {
            data: null,
            usedFallback: false,
            error: streamErr,
            aborted: true,
          };
        }
        if (!options.isFallbackEligible(streamErr)) {
          return {
            data: null,
            usedFallback: false,
            error: streamErr,
            aborted: false,
          };
        }
        if (!options.fallback) {
          return {
            data: null,
            usedFallback: true,
            error: null,
            aborted: false,
          };
        }
        try {
          const data = await options.fallback();
          return {
            data,
            usedFallback: true,
            error: null,
            aborted: false,
          };
        } catch (fallbackErr) {
          return {
            data: null,
            usedFallback: true,
            error: fallbackErr,
            aborted: false,
          };
        }
      } finally {
        setGenerating(false);
      }
    },
    [],
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { run, abort, generating };
}
