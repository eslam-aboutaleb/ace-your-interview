import { Link } from "react-router-dom";
import type { JourneyDetail, JourneyQuestionSlot } from "@/types";

interface CurriculumRailProps {
  journey: JourneyDetail;
  /** Accepted slots for the batch being read. */
  slots: JourneyQuestionSlot[];
  activeQuestionId: string | null;
}

const STATUS_DOT: Record<string, string> = {
  completed: "bg-udemy-success",
  in_progress: "bg-udemy-warning",
  pending: "bg-udemy-border",
  retired: "bg-udemy-text-muted",
};

/**
 * Curriculum rail: the journey plan (phases -> modules -> objectives) with
 * links into the batch's question slots (issue #7). Objectives without a
 * generated slot are shown but not linked.
 */
export default function CurriculumRail({
  journey,
  slots,
  activeQuestionId,
}: CurriculumRailProps) {
  const slotByObjective = new Map<string, JourneyQuestionSlot>();
  for (const slot of slots) {
    if (!slotByObjective.has(slot.objective_id)) {
      slotByObjective.set(slot.objective_id, slot);
    }
  }

  return (
    <nav aria-label="Journey curriculum" className="p-3">
      {journey.phases.map((phase) => (
        <section key={phase.id} className="mb-4">
          <h3 className="px-2 text-xs font-semibold uppercase tracking-wide text-udemy-text-muted">
            {phase.title}
          </h3>
          {phase.modules.map((module) => (
            <div key={module.id} className="mt-1">
              <p className="px-2 py-1 text-sm font-medium text-udemy-text">{module.title}</p>
              <ul>
                {module.objectives.map((objective) => {
                  const slot = slotByObjective.get(objective.id);
                  const isActive = slot != null && slot.id === activeQuestionId;
                  const row = (
                    <>
                      <span
                        aria-hidden="true"
                        className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
                          STATUS_DOT[objective.status] ?? "bg-udemy-border"
                        }`}
                      />
                      <span className="min-w-0 flex-1 truncate">{objective.title}</span>
                    </>
                  );
                  return (
                    <li key={objective.id}>
                      {slot ? (
                        <Link
                          to={`/journeys/${journey.id}/batches/${slot.batch_id}/questions/${slot.id}`}
                          aria-current={isActive ? "page" : undefined}
                          className={`flex items-start gap-2 rounded-md px-2 py-1.5 text-sm ${
                            isActive
                              ? "bg-udemy-bg font-semibold text-udemy-purple"
                              : "text-udemy-text hover:bg-udemy-bg"
                          }`}
                        >
                          {row}
                        </Link>
                      ) : (
                        <span className="flex items-start gap-2 px-2 py-1.5 text-sm text-udemy-text-muted">
                          {row}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </section>
      ))}
    </nav>
  );
}
