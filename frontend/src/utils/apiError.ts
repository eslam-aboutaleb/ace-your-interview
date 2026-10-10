/**
 * Shared API error-message extraction.
 *
 * Extracted from `InterviewSetup.tsx` so every page renders
 * server errors the same way. The FastAPI backend rejects with
 * `{ detail: string | { message, code } }`; axios surfaces it
 * as `error.response.data.detail`.
 */

export function extractApiErrorMessage(
  error: unknown,
  fallback = "Something went wrong.",
): string {
  const detail = (
    error as {
      response?: { data?: { detail?: unknown } };
      message?: string;
    }
  )?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (
    typeof detail === "object" &&
    detail &&
    typeof (detail as { message?: unknown }).message === "string" &&
    (detail as { message: string }).message.trim()
  ) {
    return (detail as { message: string }).message.trim();
  }

  const message = (error as { message?: unknown })?.message;
  if (typeof message === "string" && message.trim()) {
    return message.trim();
  }

  return fallback;
}
