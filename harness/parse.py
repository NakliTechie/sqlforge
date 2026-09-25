"""One turn-parsing rule for both harnesses (harness/loop.py for measurement, train/rollout.py for training), so
the measured object and the trained object end episodes on the same turns.

Rule, applied to each assistant turn:
1. Tool calls come from the answer; if the answer has none, from the thinking. Qwen sometimes writes its
   <tool_call> XML before closing </think>, and the intent is unambiguous.
2. With no tool call, a ```sql block in the ANSWER is a text submission. SQL written in the thinking is a plan,
   never a submission.
3. With neither, the turn is not final: the harness replies EMPTY_TURN_NOTE and the episode continues (the turn
   still counts against the cap). Before 2026-09-25 loop.py ended these episodes as NOSUBMIT, and rollout.py
   submitted SQL found in the thinking.
"""
from __future__ import annotations

import re

QWEN_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
QWEN_PARAM = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.S)
SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.S | re.I)

EMPTY_TURN_NOTE = ("No tool call was received and no final answer was given. Call run_sql to test a query, "
                   "or call submit with your one final SQL.")


def parse_calls(text: str) -> list[dict]:
    """Qwen XML tool calls → [{"name", "args"}]."""
    return [{"name": name.strip(), "args": dict(QWEN_PARAM.findall(body))} for name, body in QWEN_CALL.findall(text or "")]


def split_thinking(raw: str) -> tuple[str, str]:
    """Raw generation after a '<think>\\n' prompt → (thinking, answer). No closing tag (truncated mid-thought)
    means the whole turn is thinking."""
    if "</think>" not in raw:
        return raw, ""
    thinking, answer = raw.split("</think>", 1)
    return thinking.replace("<think>", "").strip(), answer.strip()


def sql_in_answer(answer: str) -> str | None:
    m = SQL_BLOCK.findall(answer or "")
    return m[-1].strip() if m else None
