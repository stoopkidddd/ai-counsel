"""Unit tests for deliberation.prompts pure functions."""
import pytest

from deliberation.prompts import (
    challenge_wrapper,
    pinned_question_header,
    stance_instructions,
)


class TestStanceInstructions:
    """stance_instructions returns stance-specific guidance with ethical override."""

    def test_for_stance_includes_supportive_framing(self):
        out = stance_instructions("for")
        assert "SUPPORTIVE" in out
        assert "OVERRIDE" in out  # ethical override clause

    def test_against_stance_includes_critical_framing(self):
        out = stance_instructions("against")
        assert "CRITICAL" in out
        assert "OVERRIDE" in out

    def test_neutral_stance_includes_balanced_framing(self):
        out = stance_instructions("neutral")
        assert "NEUTRAL" in out or "balanced" in out.lower()
        assert "OVERRIDE" in out

    def test_each_stance_returns_distinct_text(self):
        for_text = stance_instructions("for")
        against_text = stance_instructions("against")
        neutral_text = stance_instructions("neutral")
        assert for_text != against_text
        assert for_text != neutral_text
        assert against_text != neutral_text


class TestChallengeWrapper:
    """challenge_wrapper frames prior responses for critical evaluation."""

    def test_includes_model_name(self):
        out = challenge_wrapper("some response", "claude@opus")
        assert "claude@opus" in out

    def test_includes_response_text(self):
        out = challenge_wrapper("the response body", "x@y")
        assert "the response body" in out

    def test_instructs_critical_evaluation(self):
        out = challenge_wrapper("r", "m")
        # Must signal "don't defer" intent in some form
        assert "critical" in out.lower() or "do not defer" in out.lower()

    def test_marks_clear_start_and_end(self):
        out = challenge_wrapper("r", "m")
        # Wrapping must have visually distinct start/end markers
        assert out.count("---") >= 2


class TestPinnedQuestionHeader:
    """pinned_question_header surfaces the original question authoritatively."""

    def test_includes_question_text(self):
        q = "Should we adopt strategy X?"
        out = pinned_question_header(q)
        assert q in out

    def test_marked_authoritative(self):
        out = pinned_question_header("q")
        assert "Authoritative" in out or "authoritative" in out

    def test_warns_against_redefinition(self):
        out = pinned_question_header("q")
        # Some text indicating later context cannot redefine the question
        assert "redefine" in out.lower() or "reframe" in out.lower()
