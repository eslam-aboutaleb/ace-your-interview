import { describe, expect, it } from "vitest";
import {
  normalizeEscapedMultilineText,
  normalizeEscapedSingleLineText,
} from "./textNormalization";

describe("textNormalization", () => {
  it("decodes escaped newlines when markdown structure is present", () => {
    const input =
      "First paragraph\\n\\n## Heading\\n\\n- item one\\n- item two";

    expect(normalizeEscapedMultilineText(input)).toBe(
      "First paragraph\n\n## Heading\n\n- item one\n- item two",
    );
  });

  it("leaves text without escaped newlines untouched", () => {
    const input = "Plain text with\nreal newlines";

    expect(normalizeEscapedMultilineText(input)).toBe(input);
  });

  it("collapses escaped newlines into single spaces for single-line output", () => {
    expect(normalizeEscapedSingleLineText("one\\n\\ntwo\\nthree")).toBe(
      "one two three",
    );
  });
});
