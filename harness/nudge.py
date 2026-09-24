"""Turn-budget nudge, shared by both harnesses (harness/loop.py for measurement, train/rollout.py for
training) so the measured object and the trained object are the same. Appended to the tool observation
when the remaining budget is short; the model otherwise never sees how many turns it has left.

Motivation (2026-09-24 leg): 31 of 47 failed 2B episodes never called submit — the model kept exploring
until the cap. The nudge targets that aggregation step; it does not change the tools or the reward."""
NUDGE_TURNS = 5


def budget_note(turns_left: int) -> str:
    if turns_left > NUDGE_TURNS:
        return ""
    if turns_left <= 1:
        return "\n\n[budget] Last turn. Call submit now with your best final SQL; any other call ends the task unanswered."
    return (f"\n\n[budget] {turns_left} turns left. Stop exploring: assemble ONE final SQL from what you have "
            "verified and call submit.")
