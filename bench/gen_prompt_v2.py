"""Emit the two prompt files for deck v2, verbatim from config.py.

v1 shipped one 13 KB prompt.md. That does not fit the bottom-left quarter of a
slide, which is where the deck puts it, so the block was cut by hand when it was
pasted -- and a hand-cut prompt is a prompt that can drift from the code.

Here there are two files instead:

  prompt.md       <= QUARTER_PAGE_CHARS, the slide text. Persona, owner, clock
                  guard verbatim; the 4,279-char tools block as an excerpt with
                  its length stated.
  prompt_full.md  everything verbatim, nothing cut. The record.

Both are read out of the live module, so neither can disagree with what the
model receives. Run from the repo root:

    .venv/bin/python bench/gen_prompt_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DECK = ROOT / "ME 3 Deck v2"

# A quarter of a 1920x1080 slide is 960x540. At 9 pt in a text box that is
# ~2,200 characters before it overflows. 2,200 is the budget.
QUARTER_PAGE_CHARS = 2200


def first_sentences(text: str, n: int = 1) -> str:
    """The opening of a block, cut at a sentence end so the excerpt is honest."""
    out = ""
    for part in text.split(". "):
        cand = (out + ". " + part).strip() if out else part
        if out and cand.count(". ") >= n:
            break
        out = cand
    return out.strip() + " \u2026"


def main() -> None:
    tools_excerpt = first_sentences(config.TOOLS_SYSTEM_PROMPT, 1)
    clock = config.clock_line().strip()

    # agent.py:709-712 -- one space between every term, then .strip()
    assembled = (config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT
                 + " " + config.clock_line()
                 + " " + config.CLOCK_GUARD).strip()
    assembled = (assembled + " " + config.TOOLS_SYSTEM_PROMPT).strip()

    slide = f"""# System prompt (v3)

`SYSTEM_PROMPT + OWNER_PROMPT + clock_line() + CLOCK_GUARD [+ TOOLS_SYSTEM_PROMPT]`

**Persona** \u2014 `config.py` `SYSTEM_PROMPT`
```text
{config.SYSTEM_PROMPT}
```

**Owner** \u2014 `OWNER_PROMPT`
```text
{config.OWNER_PROMPT}
```

**Clock guard** \u2014 `CLOCK_GUARD`
```text
{config.CLOCK_GUARD}
```

**Tools** \u2014 `TOOLS_SYSTEM_PROMPT`, {len(config.TOOLS_SYSTEM_PROMPT):,} chars, only when the MCP bridge is up
```text
{tools_excerpt}
```
Full text: `prompt_full.md`. Model `{config.LLM_MODEL}` \u00b7 temp {config.LLM_TEMPERATURE} \u00b7 max {config.LLM_MAX_TOKENS} tokens \u00b7 thinking off.
"""

    full = f"""# System prompt (v3) \u2014 complete, verbatim

Every block below is copied out of `config.py` by `bench/gen_prompt_v2.py`. This is
the record; `prompt.md` is the quarter-page cut that goes on the slide.

## How it is assembled

`agent.py` composes it in this order, and sets it on the LLM service's
`system_instruction` \u2014 **not** duplicated into the context, because the copy in
the context went stale and shadowed the refreshed clock:

```
SYSTEM_PROMPT + OWNER_PROMPT + clock_line() + CLOCK_GUARD [+ TOOLS_SYSTEM_PROMPT]
```

`TOOLS_SYSTEM_PROMPT` is appended only when the MCP tool bridge is up.

## 1 \u00b7 `SYSTEM_PROMPT` ({len(config.SYSTEM_PROMPT)} chars)

```text
{config.SYSTEM_PROMPT}
```

## 2 \u00b7 `OWNER_PROMPT` ({len(config.OWNER_PROMPT)} chars)

Without this block the model answered "I don't know your name" to *"What is my
name?"*. The GitHub handle is here because `github_repo` acts on that account.

```text
{config.OWNER_PROMPT}
```

## 3 \u00b7 `clock_line()` \u2014 generated per call ({len(clock)} chars)

Sample from this machine. Generated, not baked in, so a long-running process
cannot drift to yesterday.

```text
{clock}
```

## 4 \u00b7 `CLOCK_GUARD` ({len(config.CLOCK_GUARD)} chars)

```text
{config.CLOCK_GUARD}
```

## 5 \u00b7 `TOOLS_SYSTEM_PROMPT` ({len(config.TOOLS_SYSTEM_PROMPT):,} chars)

The largest thing in the prompt by an order of magnitude. It carries the tool
list, the approval protocol, and the recency rule that stops the model calling
`search` without `type="news"`.

```text
{config.TOOLS_SYSTEM_PROMPT}
```

## The assembled string

Joined exactly as `agent.py` joins it -- every term separated by one space, the
whole thing stripped:

```text
{assembled}
```

## What is deliberately absent

No "max two sentences per turn" and no "stop and listen when the user speaks".
Turn length is bounded by `LLM_MAX_TOKENS = {config.LLM_MAX_TOKENS}`; barge-in is a
VAD-driven `InterruptionFrame`, not an instruction. Asking a model to stop
speaking while its stream is being cancelled does nothing \u2014 cancelling the
stream is what works.

## Settings that travel with the prompt

| Setting | Value |
|---|---|
| Model | `{config.LLM_MODEL}` |
| Base URL | `{config.LLM_BASE_URL}` |
| Temperature | `{config.LLM_TEMPERATURE}` |
| Max tokens | `{config.LLM_MAX_TOKENS}` |
| Thinking | {"on" if config.LLM_THINKING else "off"} |
| Tools in prompt | {len(config.TOOLS_SYSTEM_PROMPT):,} chars when the bridge is up |
"""

    (DECK / "prompt.md").write_text(slide)
    (DECK / "prompt_full.md").write_text(full)

    n = len(slide)
    print(f"prompt.md       {n:>5} chars  ({'fits' if n <= QUARTER_PAGE_CHARS else 'OVER'} "
          f"the {QUARTER_PAGE_CHARS}-char quarter page)")
    print(f"prompt_full.md  {len(full):>5} chars")


if __name__ == "__main__":
    DECK.mkdir(exist_ok=True)
    main()
