# Context engineering

## What actually goes in

| Goes in | Detail |
|---|---|
| System prompt + persona | `SYSTEM_PROMPT + OWNER_PROMPT + clock_line() + CLOCK_GUARD`, plus a 4,279-char tools block when the bridge is up. On `system_instruction`, **not** in the context |
| ASR text | The whisper final transcript. Streaming partials are published to the dashboard but do **not** enter the context |
| VAD turn events | Not context — they drive the aggregator that decides *when* to send |
| MCP tool results | Capped at 24,000 chars per result, structurally |
| Memory / RAG | **None.** No vector store, no summary, no rolling window |

## What is managed

| Managed | How |
|---|---|
| Token budget | `LLM_MAX_TOKENS = 300` caps the reply. Nothing caps the input |
| Trimming / summarising | **Nothing.** The message list grows unbounded |
| Cached prefix (TTFT) | **Nothing.** No prompt caching, no stable-prefix ordering |
| Turn states | VAD start → aggregator → LLM → TTS → audio out, one clock |
| Barge-in | `InterruptionFrame` stops TTS and cancels the LLM stream; stale audio is dropped by the output transport |
| Context size logged | Yes, per turn: `context_tokens`, `context_messages` |

## Measured, over the run

| | Value |
|---|---|
| LLM calls | 94 |
| Prompt tokens per call | median 879, max 1,674 |
| Messages in context | median 47.5, max 94 |
| First → last call | 7 → 1,674 tokens |

Per run: median 157, 514, 924 tokens — the growth is the conversation, not a leak.

## The three decisions that shaped this

**The prompt is not in the context.** Putting it in both sent it twice, and the
stale copy in the context shadowed the refreshed clock line.

**The result cap is structural.** A blind slice at 1,200 chars turned a 5-result
news payload into 2 usable results with the `date` cut. `_cap_for_model` keeps
whole results and reports how many were dropped, so a JSON list survives as a
list.

**User frames are muted during a tool call.** With no AEC the agent's own
"Searching for that." came back through the mic and cancelled the search it had
just announced — 5 of 7 cancelled calls were self-inflicted. The mute covers
exactly the function-call window and lifts on the result, so barge-in on answers
is untouched.

## The failure mode to know about

**Untrimmed context fails silently.** Past the endpoint's ceiling the request
returns 400, which arrives as an `ErrorFrame` that is logged and pushed to the
dashboard — but **nothing is spoken**. In a voice-only demo a failed turn is
indistinguishable from a dead agent. A trim plus one spoken line ("that got long
— starting fresh") would convert a silent hang into a recoverable turn.
