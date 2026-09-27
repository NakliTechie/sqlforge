"""Multi-turn rollouts for training: build the token sequence segment by segment so the loss
mask lands exactly on model-generated tokens; tool observations stay in context, masked out.

A policy needs `generate_batch(prompts, seeds) -> texts` (raw text continuations of raw prompts). HFPolicy
(train/rollout.py) generates one sequence at a time; VLLMPolicy (train/vllm_policy.py) batches every active episode
into one call with prefix caching. `run_episodes` advances many episodes in lockstep, one batched call per turn;
`run_episode` is the single-episode case of the same state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from harness.nudge import budget_note
from harness.parse import EMPTY_TURN_NOTE, parse_calls, split_thinking, sql_in_answer

EOS = "<|im_end|>"


@dataclass
class Trajectory:
    segments: list[tuple[str, bool]] = field(default_factory=list)  # (text, generated?)
    turns: int = 0
    n_tool: int = 0
    submitted_sql: str | None = None
    hit_cap: bool = False
    context_overflow: bool = False  # ended because the next turn would not fit the model context
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

    def generate_batch(self, prompts: list[str], seeds: list[int | None]) -> list[str]:
        return [self.generate(p) for p in prompts]


def reply_chunk(tok, role: str, content: str, think: bool) -> str:
    """The template's text between an assistant turn's <|im_end|> and the next generation point, for a tool
    observation (role="tool") or a harness note (role="user")."""
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": role, "content": content}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return full[full.index("y<|im_end|>") + len("y<|im_end|>"):]


class Episode:
    """One rollout as a state machine: prompt() → policy text → advance(text). done() once it submits or caps."""

    def __init__(self, tok, system: str, question: str, tools: list, run_sql, *, max_turns: int, think: bool = True,
                 seed: int | None = None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=think)
        self.tok, self.run_sql, self.max_turns, self.think, self.seed = tok, run_sql, max_turns, think, seed
        self.tr = Trajectory(segments=[(prompt, False)])
        self._done = False

    def done(self) -> bool:
        return self._done

    def prompt(self) -> str:
        return "".join(t for t, _ in self.tr.segments)

    def advance(self, text: str) -> None:
        tr = self.tr
        tr.turns += 1
        if EOS not in text:
            text = text + EOS  # truncated generation: close the turn so the sequence stays well-formed
        text = text[: text.index(EOS) + len(EOS)]
        tr.segments.append((text, True))
        tr.final_text = text
        body = text[: -len(EOS)]
        thinking, answer = split_thinking(body) if self.think else ("", body)
        calls = parse_calls(answer) or parse_calls(thinking)                   # harness/parse.py rule 1
        left = self.max_turns - tr.turns
        if not calls:
            sql = sql_in_answer(answer)                                        # rule 2
            if sql:
                tr.submitted_sql, self._done = sql, True
                return
            tr.segments.append((reply_chunk(self.tok, "user", EMPTY_TURN_NOTE + budget_note(left), self.think), False))
        else:
            obs_parts = []
            for c in calls:
                tr.n_tool += 1
                if c["name"] == "submit":
                    tr.submitted_sql, self._done = c["args"].get("sql", ""), True
                    return
                if c["name"] == "run_sql":
                    cols, rows, err = self.run_sql(c["args"].get("sql", ""))
                    obs_parts.append(_fmt(cols, rows, err))
                else:
                    obs_parts.append(f"ERROR: unknown tool {c['name']}")
            tr.segments.append((reply_chunk(self.tok, "tool", "\n\n".join(obs_parts) + budget_note(left), self.think), False))
        if tr.turns >= self.max_turns:
            tr.hit_cap, self._done = True, True


def run_episodes(policy, episodes: list[Episode]) -> list[Trajectory]:
    """Advance all episodes in lockstep: one batched generate per turn over the ones still running.
    Context guard: an episode whose prompt no longer leaves room for a full turn (policy.max_model_len, when the policy
    has one) ends as a turn-cap, i.e. no submission. Real schemas + 25 thinking turns can exceed 32k tokens; on
    2026-09-27 one such prompt raised vLLM's VLLMValidationError and killed climb-2 life 7 at step 50 (10 steps lost)."""
    limit = getattr(policy, "max_model_len", None)
    while True:
        live = [e for e in episodes if not e.done()]
        if limit:
            for e in live:
                if len(e.tok(e.prompt()).input_ids) + policy.max_new_tokens > limit:
                    e.tr.hit_cap, e.tr.context_overflow, e._done = True, True, True
            live = [e for e in live if not e.done()]
        if not live:
            return [e.tr for e in episodes]
        texts = policy.generate_batch([e.prompt() for e in live], [e.seed for e in live])
        for e, text in zip(live, texts):
            e.advance(text)


def run_episode(policy, tok, system: str, question: str, tools: list, run_sql, *, max_turns: int,
                think: bool = True) -> Trajectory:
    return run_episodes(policy, [Episode(tok, system, question, tools, run_sql, max_turns=max_turns, think=think)])[0]


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
