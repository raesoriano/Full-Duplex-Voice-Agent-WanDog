"""Emit "ME 3 Deck/prompt.md" with the system prompt copied verbatim from config.py.

The point is that the deck cannot drift from the code: every block below is read
out of the live module, not retyped. Run from the repo root:

    .venv/bin/python bench/gen_prompt_md.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck" / "prompt.md"

HEAD = """# System prompt (v3) — as composed at runtime

WanDog is the voice agent. This is the **actual** prompt, not a summary of it.
Every block below is copied verbatim from `config.py` by `bench/gen_prompt_md.py`,
so what the deck shows is what the model receives.

## How it is assembled

`agent.py:709` builds the prompt from four constants and one call, in this order:

```
SYSTEM_PROMPT + OWNER_PROMPT + clock_line() + CLOCK_GUARD [+ TOOLS_SYSTEM_PROMPT]
```

The last term is conditional: `TOOLS_SYSTEM_PROMPT` is appended only when the MCP
tool bridge is up (`MCP_TOOLS_ENABLED`, default on). With tools off the agent
still gets the persona, the owner block, the clock and the clock guard — it just
has no tools to call.

The prompt is set on the LLM service's `system_instruction`, **not** duplicated
into the context (`agent.py:838`): putting it in both sent it twice, and the
stale copy in the context would have shadowed the refreshed clock.

---
"""

BODY = f"""## 1 · Persona — `config.py:517` `SYSTEM_PROMPT`

```text
{config.SYSTEM_PROMPT}
```

## 2 · Who it is talking to — `config.py:524` `OWNER_PROMPT`

```text
{config.OWNER_PROMPT}
```

The owner block exists because without it the model answered "I don't know your
name" to *"What is my name?"* on the live pipeline. The GitHub handle is in there
because `github_repo` and `git` both act on that account, so "what did I push
last?" needs the model to know whose repos "mine" means.

## 3 · The clock — `config.py:clock_line()`

Generated at call time, not baked in, so a long-running process cannot drift to
yesterday's date. A sample, taken from this machine:

```text
{config.clock_line().strip()}
```

## 4 · Clock guard — `config.py` `CLOCK_GUARD`

```text
{config.CLOCK_GUARD}
```

Always present, even with tools off: an agent with no tools has a clock in its
prompt but no `date` to run, and it still must not fill the gap from memory.

## 5 · Tools — `config.py:564` `TOOLS_SYSTEM_PROMPT`

Appended only when the MCP bridge is active. {len(config.TOOLS_SYSTEM_PROMPT):,} characters, which is
why it is the single largest thing in the prompt.

```text
{config.TOOLS_SYSTEM_PROMPT}
```

---

## The assembled prompt

Concatenated exactly as `agent.py` does it, with the clock fixed at the sample
above. This is the string handed to the LLM as `system_instruction`.

```text
{config.SYSTEM_PROMPT} {config.OWNER_PROMPT} {config.clock_line()}{config.CLOCK_GUARD} {config.TOOLS_SYSTEM_PROMPT}
```

## LLM settings that travel with it

| Setting | Value | Where |
|---|---|---|
| Model | `{config.LLM_MODEL}` | `config.py:235` |
| Base URL | `{config.LLM_BASE_URL}` | `config.py:234` |
| Temperature | `{config.LLM_TEMPERATURE}` | `config.py:236` |
| Max tokens | `{config.LLM_MAX_TOKENS}` | `config.py:237` |
| Thinking | `{"on" if config.LLM_THINKING else "off (enable_thinking=False)"}` | `config.py:276` |

Thinking is disabled because reasoning models return an empty `content` field by
default — their output goes to a separate reasoning field, so the voice agent
received nothing to speak. The flag rides in `extra_body` as a vLLM-style
`chat_template_kwargs`.

## What is deliberately absent

No "max 2 sentences per turn" rule, and no "stop and listen on barge-in"
instruction. Both are enforced by the **pipeline**, not the prompt: turn length
is capped by `LLM_MAX_TOKENS = {config.LLM_MAX_TOKENS}`, and barge-in is a VAD-driven
`InterruptionFrame` (`agent.py:797-801`). Asking the model to stop talking while
it is being interrupted does not work; cancelling its stream does.
"""

OUT.write_text(HEAD + BODY, encoding="utf-8")
print(f"wrote {OUT} ({len(HEAD + BODY)} chars)")
