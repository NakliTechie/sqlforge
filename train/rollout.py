"""Multi-turn rollouts for training: build the token sequence segment by segment so the loss
mask lands exactly on model-generated tokens; tool observations stay in context, masked out.

A Policy only needs `generate(prompt_text) -> text`. HFPolicy is local (laptop smoke, single-GPU
climb); a vLLM/SGLang policy plugs in with the same one method.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

from harness.nudge import budget_note

QWEN_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
QWEN_PARAM = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text: str) -> list[dict]:
    out = []
    for name, body in QWEN_CALL.findall(text):
        args = {k: v for k, v in QWEN_PARAM.findall(body)}
        out.append({"name": name.strip(), "args": args})
    return out


@dataclass
class Trajectory:
    segments: list[tuple[str, bool]] = field(default_factory=list)  # (text, generated?)
    turns: int = 0
    n_tool: int = 0
    submitted_sql: str | None = None
    hit_cap: bool = False
    final_text: str = ""

    def gen_chars(self) -> int:
        return sum(len(t) for t, g in self.segments if g)


class HFPolicy:
    def __init__(self, model, tokenizer, device, max_new_tokens=512, temperature=0.6):
        self.model, self.tok, self.device = model, tokenizer, device
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    @torch.no_grad()
    def generate(self, prompt_text: str) -> str:
        ids = self.tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=True,
                                  temperature=self.temperature, top_p=0.95,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        gen = out[0, ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=False)


def tool_response_chunk(tok, obs: str) -> str:
    """The template's text between an assistant turn's <|im_end|> and the next generation point."""
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "tool", "content": obs}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return full[full.index("y<|im_end|>") + len("y<|im_end|>"):]


def run_episode(policy: HFPolicy, tok, system: str, question: str, tools: list, run_sql, *, max_turns: int) -> Trajectory:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    tr = Trajectory(segments=[(prompt, False)])
    eos = "<|im_end|>"
    while tr.turns < max_turns:
        tr.turns += 1
        text = policy.generate("".join(t for t, _ in tr.segments))
        if eos not in text:
            text = text + eos  # truncated generation: close the turn so the sequence stays well-formed
        text = text[: text.index(eos) + len(eos)]
        tr.segments.append((text, True))
        tr.final_text = text
        calls = parse_calls(text)
        if not calls:
            m = re.findall(r"```sql\s*(.*?)```", text, flags=re.S | re.I)
            tr.submitted_sql = m[-1].strip() if m else None
            return tr
        obs_parts = []
        for c in calls:
            tr.n_tool += 1
            if c["name"] == "submit":
                tr.submitted_sql = c["args"].get("sql", "")
                return tr
            if c["name"] == "run_sql":
                cols, rows, err = run_sql(c["args"].get("sql", ""))
                obs_parts.append(_fmt(cols, rows, err))
            else:
                obs_parts.append(f"ERROR: unknown tool {c['name']}")
        tr.segments.append((tool_response_chunk(tok, "\n\n".join(obs_parts) + budget_note(max_turns - tr.turns)), False))
    tr.hit_cap = True
    return tr


def _fmt(cols, rows, err, max_rows=20, max_chars=1500):
    if err:
        return f"ERROR: {err}"
    if not rows:
        return f"columns: {cols}\n(0 rows)"
    body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:max_rows])
    tail = f"\n... ({len(rows)} rows total, showing {max_rows})" if len(rows) > max_rows else f"\n({len(rows)} rows)"
    return (" | ".join(cols) + "\n" + body + tail)[:max_chars]


def tokenize_trajectory(tok, tr: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-segment token ids; mask=1 on generated tokens."""
    ids, mask = [], []
    for text, gen in tr.segments:
        seg = tok(text, add_special_tokens=False).input_ids
        ids.extend(seg)
        mask.extend([1 if gen else 0] * len(seg))
    return torch.tensor(ids), torch.tensor(mask)
