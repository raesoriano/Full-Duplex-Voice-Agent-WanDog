# Context engineering — what actually goes in, and what is managed

WanDog · Pipecat 1.15.0 · Raspberry Pi 5 (8 GB)

The short version: **almost nothing is managed.** The context is an unbounded
message list that grows for the life of the session. What this document does is
separate the parts that are deliberate from the parts that are simply absent, and
give the measured numbers for both.

---

## GOES IN

### The system prompt — once, not twice

`SYSTEM_PROMPT` + `OWNER_PROMPT` + `clock_line()` + `CLOCK_GUARD` +
`TOOLS_SYSTEM_PROMPT` ≈ **5.6 k characters (~1.4 k tokens)**. It rides on the LLM
service's `system_instruction`, **not** in the context (`agent.py:838`).

That is a fix, not a style choice. Putting it in both places sent every request
with two identical ~1,400-character system messages, and the stale copy in the
context would have shadowed any refresh of the live one — which is exactly what
the clock needs. Pipecat warns about this and calls the context copy deprecated.

### ASR text — final only, in the context

Only the **final** transcript enters the context, as a user message. Interim
partials are published to the dashboard over the event bus
(`PartialTranscriptTap`, `agent.py:884`) and **never** sent to the LLM. That keeps
the context from filling with the same sentence re-spelled six ways.

### VAD turn events — not in the context, but they shape it

VAD frames are not messages. They act on the *boundary* of a message: a
`VADUserStoppedSpeakingFrame` flushes the ASR segment, and a
`VADUserTurnStartStrategy` start raises an `InterruptionFrame` that ends the turn.
So the VAD decides *when* a message exists, not what it says.

### MCP tool results — capped at 24,000 characters each

`MCP_MODEL_RESULT_CHARS = 24000` (`config.py:360`). The cap is applied
**structurally** by `mcp_bridge._cap_for_model`: a JSON list keeps whole results
and says how many were dropped, rather than being sliced mid-string. A blind
600-character slice had turned a 5-result news payload into 1.6 broken results —
the model got a title and a dangling fragment, and could not see the `date` field
it needed to tell old news from new.

The full result is still kept for the dashboard and the session folder; only the
model's copy is capped.

Why 24,000 and not 1,200: measured on 17 Nov 2026, **every** shape the tools
return was being cut at 1,200, and two lost the answer rather than the bulk.

| Payload | Raw size | at 1,200 chars | at 24,000 chars |
|---|---|---|---|
| ddgs news, 5 results | 2,073 | 2 of 5 | 5 of 5 |
| Tavily news, 5 results | 6,694 | 2 of 5 | 5 of 5 |
| web search, 5 page bodies | 7,432 | 2 of 5 | 5 of 5 |
| ollama web, 5 results | 22,970 | 2 of 5 | 5 of 5 |
| `read_file` | 12,094 | clipped | whole |

### Memory / RAG — none

There is no vector store, no summarisation memory, no cross-session recall. Each
session starts from an empty `LLMContext(messages=[])`. `local_search` is a *tool*
the model may call, not a retrieval step in the pipeline — nothing is retrieved
unless the model decides to ask.

---

## MANAGED

| Concern | State | Evidence |
|---|---|---|
| Token budget | **Not managed** — no trimming, no summarisation | no `trim`/`summarize`/`max_turns` in `agent.py`, `config.py`, `mcp_bridge.py` |
| Stable prefix caching | **Not done** — no cache hints sent | no `cache_control` anywhere |
| Tool result size | Managed — hard 24,000-char cap | `config.py:360`, `mcp_bridge._cap_for_model` |
| Turn length | Managed — `LLM_MAX_TOKENS = 300` | `config.py:237` |
| Barge-in stale audio | Managed — `InterruptionFrame` broadcast both directions | `agent.py:797-801` |
| Context size per run | **Measured**, logged per turn | `context_tokens` / `context_messages` in `stage_latency.json` |

### The token budget is the gap

`context_kwargs = {"messages": []}` (`agent.py:845`) and nothing ever prunes it.
The benchmark measured the consequence over a 40-trial session:

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate) | 879 | 1,674 | 7 → 1,674 |
| Messages in context | 48 | 94 | 1 → 94 |

It never hit a limit, because the session was not long enough and the endpoint
accepts ~1.5 M characters. It is still unbounded: a long conversation grows
without check until the endpoint returns HTTP 400, which reaches the pipeline as
an `ErrorFrame` that `agent.py` only logs — so the turn goes **silent** instead of
short. That is the failure mode to know about.

### Idle → listen → think → speak

Not a state machine in this agent. The turn cycle is Pipecat's: VAD start
debounces for `VAD_START_SECS = 0.25` s at `confidence 0.6` and `min_volume 0.35`,
the ASR streams throughout, and the turn closes `TURN_STOP_SECS = 1.0` s after the
VAD stop. There is no explicit idle state.

### On barge-in: stale audio is dropped, and one thing is deliberately suppressed

Speech start → `InterruptionFrame` → TTS playback cancelled and the LLM stream
cancelled (`agent.py:797`). Measured: **0.48 s** from speech to playback stopped,
against a 0.80 s target.

One asymmetry worth stating, because it is a design decision rather than an
oversight: `FunctionCallUserMuteStrategy` (`agent.py:820`) suppresses *user* frames
— VAD frames included — for exactly as long as a tool call is in flight. Without
it, the agent's own spoken acknowledgement ("Searching for that.") came back
through the mic, raised a VAD turn start, and cancelled the search it had just
announced: **5 of 7 cancelled tool calls** were cancelled by the agent's own
voice. Barge-in on the agent's *answers* is untouched; the mute lifts on
`FunctionCallResultFrame`.

That works only because there is **no AEC** in this pipeline. The mic hears the
speaker. The mute is the substitute.

---

## What this means in practice

The prompt is the largest fixed cost at ~1.4 k tokens, and it is paid on every
single call — that is the price of a fresh clock and a tool contract the model
actually follows. Tool results dominate everything else when they appear, which is
why the 24,000-character cap is the one budget that is genuinely enforced. And the
message list is the one that is not: it grows, it is measured, and it is the first
thing to fix before a session runs long enough to matter.
