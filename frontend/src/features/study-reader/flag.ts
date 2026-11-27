/**
 * Study reader feature flag (issue #7). Default OFF: the new routes only
 * register when VITE_STUDY_READER_V1=true, so the legacy TopicStudy flow
 * stays the sole reader until the new one is deliberately enabled.
 */
export function isStudyReaderEnabled(): boolean {
  return import.meta.env.VITE_STUDY_READER_V1 === "true";
}
