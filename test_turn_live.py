"""Live multi-turn test: does the model answer the CURRENT question?

Reproduces the reported symptom against the real endpoint, using the history
shape the broken pipeline actually produced: a growing stack of user turns
with NO assistant replies.

Before the shared-context fix, the prompt for the Nth turn was

    [system, user(Q1), user(Q2), ..., user(QN)]

Every earlier question sat there as an unanswered request. The model then
picks whichever it likes -- which is the "it answers the previous question"
symptom. After the fix it is

    [system, user(Q1), assistant(A1), ..., user(QN)]

Run:  .venv/bin/python test_turn_live.py
"""

from __future__ import annotations

import asyncio

import config

SYSTEM = (
    "You are Fweddie, a friendly voice assistant. Answer the user's most "
    "recent message only. One short sentence."
)

# A conversation whose turns are easy to tell apart by their answers.
SCRIPT = [
    ("What is 2 plus 2?", "four"),
    ("What is the capital of France?", "paris"),
    ("What colour is the sky on a clear day?", "blue"),
    ("How many legs does a spider have?", "eight"),
    ("What is the largest planet in our system?", "jupiter"),
]

FINAL_Q = "What is the boiling point of water in Celsius at sea level?"
FINAL_EXPECT = "100"


async def complete(history: list[dict], question: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=config.resolve_llm_api_key(),
        base_url=config.LLM_BASE_URL,
    )
    msgs = list(history) + [{"role": "user", "content": question}]
    r = await client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=msgs,
        temperature=0.0,
        max_tokens=120,
    )
    return (r.choices[0].message.content or "").strip()


def addresses(reply: str, expect: str) -> bool:
    return expect.lower() in reply.lower()


async def main() -> int:
    print(f"\nLLM : {config.LLM_MODEL} @ {config.LLM_BASE_URL}")
    print(f"final question: {FINAL_Q!r}  (expect {FINAL_EXPECT!r})")

    # ---------- A: the broken shape -- no assistant turns ----------
    broken: list[dict] = [{"role": "system", "content": SYSTEM}]
    for q, _ in SCRIPT:
        broken.append({"role": "user", "content": q})

    print(f"\n=== A: {len(SCRIPT)} stacked user turns, NO replies (the old bug) ===")
    print(f"  prompt roles : {[m['role'] for m in broken]} + user")
    a = await complete(broken, FINAL_Q)
    print(f"  reply        : {a!r}")
    a_ok = addresses(a, FINAL_EXPECT)

    # ---------- B: the fixed shape -- replies interleaved ----------
    fixed: list[dict] = [{"role": "system", "content": SYSTEM}]
    for (q, expect), reply in zip(SCRIPT, [f"The answer is {e}." for _, e in SCRIPT]):
        fixed.append({"role": "user", "content": q})
        fixed.append({"role": "assistant", "content": reply})

    print(f"\n=== B: same turns WITH replies (after the fix) ===")
    print(f"  prompt roles : {[m['role'] for m in fixed]} + user")
    b = await complete(fixed, FINAL_Q)
    print(f"  reply        : {b!r}")
    b_ok = addresses(b, FINAL_EXPECT)

    print("\n" + "=" * 62)
    print(f"  A (no replies)     answers the live turn : {a_ok}")
    print(f"  B (with replies)   answers the live turn : {b_ok}")
    print()
    if b_ok:
        print("  PASS: with the shared context the model tracks the live turn.")
    else:
        print("  FAIL: even with replies the model loses the live turn.")
    if not a_ok:
        print("  NOTE: the broken shape missed the live turn -- symptom reproduced.")
    return 0 if b_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
