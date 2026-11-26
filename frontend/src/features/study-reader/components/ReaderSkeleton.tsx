/** Loading placeholder for the reader shell (issue #7). */
export default function ReaderSkeleton() {
  return (
    <div
      data-testid="reader-skeleton"
      aria-busy="true"
      aria-label="Loading reader"
      className="flex h-full animate-pulse"
    >
      <div className="hidden md:block w-72 border-r border-udemy-border p-4 space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-4 rounded bg-udemy-border" />
        ))}
      </div>
      <div className="flex-1 p-6 space-y-4">
        <div className="h-6 w-2/3 rounded bg-udemy-border" />
        <div className="h-4 w-full rounded bg-udemy-border" />
        <div className="h-4 w-5/6 rounded bg-udemy-border" />
        <div className="h-40 w-full rounded bg-udemy-border" />
      </div>
    </div>
  );
}
