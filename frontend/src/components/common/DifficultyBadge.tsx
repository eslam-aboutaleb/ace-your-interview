interface Props {
  difficulty: "easy" | "medium" | "hard";
}

export default function DifficultyBadge({ difficulty }: Props) {
  const cls =
    difficulty === "easy"
      ? "badge-easy"
      : difficulty === "hard"
        ? "badge-hard"
        : "badge-medium";

  return <span className={cls}>{difficulty}</span>;
}
