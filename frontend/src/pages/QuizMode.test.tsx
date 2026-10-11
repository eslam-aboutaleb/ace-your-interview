import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import QuizMode from "./QuizMode";
import { fetchTopic, fetchTopics } from "@/services/api";
import { useQuizStore } from "@/store/quizStore";
import type { QuizQuestionV2, TopicSection } from "@/types";

const highlightChatProps: Array<{
  sectionTitle?: string;
  sourceQuote?: string;
}> = [];

vi.mock("@/components/common/WordHighlightChat", () => ({
  default: (props: {
    sectionTitle?: string;
    sourceSection?: string;
    sourceQuote?: string;
    children?: React.ReactNode;
  }) => {
    highlightChatProps.push({
      sectionTitle: props.sectionTitle,
      sourceQuote: props.sourceQuote,
    });
    return <div data-testid="word-highlight-chat">{props.children}</div>;
  },
}));

vi.mock("@/components/common/MarkdownRenderer", () => ({
  default: ({ content }: { content: string }) => (
    <div data-testid="markdown">{content}</div>
  ),
}));

vi.mock("@/services/api", () => ({
  fetchTopics: vi.fn(),
  fetchTopic: vi.fn(),
  generateQuizV2Stream: vi.fn(),
  isQuizStreamError: vi.fn(),
  recordLearningAttempt: vi.fn(),
  fetchWeakAreas: vi.fn(),
  fetchHint: vi.fn(),
  fetchNextObjectives: vi.fn(),
}));

const SECTIONS: TopicSection[] = [
  {
    heading: "Rendering Pipeline",
    content: "Reconciliation compares trees before touching the DOM.",
  },
  {
    heading: "Effects",
    content: "Effects run after the browser paints the committed tree.",
  },
];

function question(overrides: Partial<QuizQuestionV2> = {}): QuizQuestionV2 {
  return {
    question_id: "q-1",
    topic_id: "frontend-react",
    question: "Why does React diff trees before updating the DOM?",
    type: "mcq",
    choices: [
      { label: "A", text: "To compute a minimal update" },
      { label: "B", text: "To slow rendering down" },
    ],
    correct_answer: "A",
    explanation: "Reconciliation compares trees to compute minimal DOM updates.",
    difficulty: "medium",
    source_quote: "Reconciliation compares trees before touching the DOM.",
    reasoning_summary: "",
    learning_objective: "",
    expected_answer: "",
    distractor_rationales: [],
    ...overrides,
  };
}

function renderQuiz() {
  return render(
    <MemoryRouter>
      <QuizMode />
    </MemoryRouter>,
  );
}

describe("QuizMode highlight chat grounding", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    highlightChatProps.length = 0;
    useQuizStore.getState().reset();
    vi.mocked(fetchTopics).mockResolvedValue([]);
    vi.mocked(fetchTopic).mockResolvedValue({
      id: "frontend-react",
      title: "React Rendering",
      description: "How React commits updates",
      track: "frontend",
      levels: ["mid"],
      sections: SECTIONS,
      response_detail: "very_detailed",
      selected_language: "",
      requires_programming: false,
      language_options: [],
    } as never);
  });

  it("passes the handbook section the quiz question was drawn from", async () => {
    useQuizStore.setState({
      phase: "playing",
      questions: [question()],
      currentIdx: 0,
    });

    renderQuiz();

    await waitFor(() => expect(highlightChatProps.length).toBeGreaterThan(0));
    await waitFor(() =>
      expect(highlightChatProps.at(-1)).toEqual({
        sectionTitle: "Rendering Pipeline",
        sourceQuote: "Reconciliation compares trees before touching the DOM.",
      }),
    );
    expect(await screen.findByTestId("word-highlight-chat")).toBeInTheDocument();
  });

  it("leaves the section empty when no section carries the quote", async () => {
    useQuizStore.setState({
      phase: "playing",
      questions: [
        question({
          source_quote: "A quotation that appears in no section at all.",
        }),
      ],
      currentIdx: 0,
    });

    renderQuiz();

    await waitFor(() =>
      expect(highlightChatProps.at(-1)?.sectionTitle).toBe(""),
    );
    expect(
      highlightChatProps.at(-1)?.sourceQuote,
    ).toBe("A quotation that appears in no section at all.");
  });

  it("passes no section when the question has no source quote", async () => {
    useQuizStore.setState({
      phase: "playing",
      questions: [question({ source_quote: "" })],
      currentIdx: 0,
    });

    renderQuiz();

    await waitFor(() => expect(highlightChatProps.length).toBeGreaterThan(0));
    expect(highlightChatProps.at(-1)?.sectionTitle).toBe("");
    expect(highlightChatProps.at(-1)?.sourceQuote).toBe("");
  });
});
