from __future__ import annotations

import string
from typing import Sequence

PROBE_WRAPPER = "{context}\n\nQuestion: {question}\n\nAnswer: "

PROBE_QUESTIONS = {
    "imperative": {
        "long_term": (
            "Examine this viewer's full movie rating record. "
            "Does the viewer enjoy {category} films?"
        ),
        "short_term": (
            "Look at the viewer's latest watching session. "
            "Would the viewer choose {category} movies next?"
        ),
    },
    "judgment": {
        "long_term": (
            "Given the audience member's complete movie history, "
            "judge whether they have a preference for {category} movies. Yes or No?"
        ),
        "short_term": (
            "Given the audience member's recent activity, "
            "judge whether {category} movies fit their next session. Yes or No?"
        ),
    },
}

CHOICE_QUESTIONS = {
    "imperative": {
        "long_term": (
            "Examine this viewer's full movie rating record. "
            "Which genre does the viewer enjoy MOST?"
        ),
        "short_term": (
            "Look at the viewer's latest watching session. "
            "Which genre will the viewer most likely choose NEXT?"
        ),
    },
    "judgment": {
        "long_term": (
            "Given the audience member's complete movie history, "
            "judge which genre is their strongest preference."
        ),
        "short_term": (
            "Given the audience member's recent activity, "
            "judge which genre best fits their next session."
        ),
    },
}

RANKING_QUESTIONS = {
    "imperative": {
        "long_term": (
            "Examine this viewer's full movie rating record. "
            "Rank the top {k} genres the viewer enjoys most."
        ),
        "short_term": (
            "Look at the viewer's latest watching session. "
            "Predict the top {k} genres the viewer will choose NEXT."
        ),
    },
    "judgment": {
        "long_term": (
            "Given the audience member's complete movie history, "
            "list the {k} strongest genre preferences in order."
        ),
        "short_term": (
            "Given the audience member's recent activity, "
            "list the {k} genres that best fit their next session in order."
        ),
    },
}

CHOICE_SUFFIX = "Answer with the letter only (A, B, C, etc.):"


def answer_letters(categories: Sequence[str]) -> list[str]:
    if len(categories) > len(string.ascii_uppercase):
        raise ValueError("more than 26 categories: use hierarchical probing")
    return list(string.ascii_uppercase[: len(categories)])


def options_text(categories: Sequence[str]) -> str:
    letters = answer_letters(categories)
    return "\n".join(f"{letter}) {category}" for letter, category in zip(letters, categories))


def probe_prompt(context: str, category: str, question: str) -> str:
    return PROBE_WRAPPER.format(context=context, question=question.format(category=category))


def choice_prompt(context: str, categories: Sequence[str], question: str) -> str:
    return f"{context}\n\n{question}\n\n{options_text(categories)}\n\n{CHOICE_SUFFIX}"


def ranking_prompt(context: str, categories: Sequence[str], question: str, k: int) -> str:
    body = question.format(k=k)
    if k == 1:
        tail = "Output only one letter:"
    else:
        tail = f"Output only {k} letters separated by spaces:"
    return f"{context}\n\n{body} Choose from:\n{options_text(categories)}\n\n{tail}"
