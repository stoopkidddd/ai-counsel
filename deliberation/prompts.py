"""Pure prompt-fragment builders used by the deliberation engine.

These functions are dependency-free and side-effect-free so they can be unit-tested
in isolation. The engine imports them and composes the final prompts.

References pal-mcp-server/tools/consensus.py for stance wording (ethical-override
guardrails) but keeps the text compact for ai-counsel's prompt budget.
"""

from typing import Literal

Stance = Literal["for", "against", "neutral"]


def stance_instructions(stance: Stance) -> str:
    """Return a stance instruction block to prepend to a participant's prompt.

    Each stance includes a mandatory ethical override: stance is a tool for
    surfacing perspectives, not a license for bad-faith reasoning.
    """
    if stance == "for":
        return (
            "## Your Stance: SUPPORTIVE (with ethical override)\n"
            "Advocate FOR this proposal — but only if it has genuine merit.\n"
            "- Identify real strengths, synergies, and viable implementation paths.\n"
            "- Propose solutions to legitimate challenges.\n"
            "MANDATORY OVERRIDE: If the proposal is harmful, infeasible, or fundamentally\n"
            "flawed, say so directly. Truth outranks advocacy. Do not support bad ideas.\n"
        )
    if stance == "against":
        return (
            "## Your Stance: CRITICAL (with fairness override)\n"
            "Critique this proposal rigorously — but only where critique is warranted.\n"
            "- Identify real risks, failure modes, and overlooked complexities.\n"
            "- Question assumptions and suggest stronger alternatives.\n"
            "MANDATORY OVERRIDE: If the proposal is genuinely sound, acknowledge it clearly\n"
            "and offer constructive refinements rather than manufactured objections.\n"
        )
    # neutral
    return (
        "## Your Stance: NEUTRAL (with truth override)\n"
        "Provide balanced analysis weighing both sides by their actual evidence.\n"
        "- Present significant pros and cons proportionally to their impact.\n"
        "MANDATORY OVERRIDE: Do not manufacture artificial 50/50 balance. If evidence\n"
        "strongly favors one conclusion, state that directly.\n"
    )


def challenge_wrapper(prior_response_text: str, model_name: str) -> str:
    """Wrap a prior round response in critical-evaluation framing.

    Used in challenge_mode to reduce sycophantic agreement: the next model sees
    the prior response framed as something to scrutinize rather than defer to.
    """
    return (
        f"--- Prior response from {model_name} (evaluate critically; do not defer) ---\n"
        f"{prior_response_text}\n"
        f"--- End prior response. Identify gaps, unstated assumptions, or weaknesses "
        f"before forming your own view. ---"
    )


def pinned_question_header(question: str) -> str:
    """Return the canonical pinned-question header.

    Always rendered at the top of every round's prompt so the original question
    cannot be diluted or reframed by accumulated context.
    """
    return (
        "## Original Question (Authoritative)\n"
        f"{question}\n\n"
        "_This is the question you are answering. Later context is reference "
        "material only and does not redefine the question._\n"
    )
