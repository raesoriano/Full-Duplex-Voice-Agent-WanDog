# Context engineering

## What actually goes in

| Goes in | Detail |
|---|---|
| System prompt + persona | `SYSTEM_PROMPT + OWNER_PROMPT + clock_line() + CLOCK_GUARD`, plus a 3,404-char tools block when the bridge is up (4,140 assembled). On `system_instruction`, **not** in the context |
| ASR text | The whisper final transcript. Streaming partials are published to the dashboard but do **not** enter the context |
| VAD turn events | Not context — they drive the aggregator that decides *when* to send |
| MCP tool results | Capped at 8,000 chars per result, structurally, with a ladder that degrades instead of collapsing |
| Memory / RAG | **None.** No vector store, no summary, no rolling window |

## What is managed

Two hands are on this, and it is worth being explicit about which: some of it
is our code, some of it is Pipecat's aggregator doing it for us.

| Managed | By | How |
|---|---|---|
| Prompt kept out of the message list | **us** | `system_instruction` on the LLM service, `agent.py:838`. Putting it in both sent it twice and the stale copy shadowed the refreshed clock |
| Clock rewritten every inference | **us** | `ClockedLLMService._refresh_clock`, `agent.py:493-520` |
| Tool results capped structurally | **us** | `_cap_for_model` keeps whole records and states how many were dropped, so a JSON list survives as a list |
| Passage chosen around the query | **us** | `_cap_best_window` keeps the opening *and* the densest passage for the question, not the page head |
| News over-fetch → sort → trim | **us** | `_shape_search_args` sets `type=news` and over-fetches; `_sort_news` orders newest-first before the cap trims |
| User frames muted during a tool call | **us** | `FunctionCallUserMuteStrategy` covers exactly the function-call window, so the agent's own "Searching for that." cannot cancel the search it announced |
| Turn boundaries and barge-in | **Pipecat** | `LLMUserAggregator` runs the VAD turn-start and speech-timeout strategies; `InterruptionFrame` stops TTS and cancels the stream |
| Assistant turn aggregation | **Pipecat** | The aggregator accumulates tokens and writes one assistant message per turn — the reason the context is a conversation, not a token soup |
| Interrupted turns recorded | **Pipecat** | `_trigger_assistant_turn_stopped(interrupted=True)` still broadcasts the partial text (`llm_response_universal.py:2735`), so a barge-in leaves a **truncated** assistant message, not a dropped one |
| Manual reset | **us → Pipecat** | `_handle_clear` pushes `LLMMessagesUpdateFrame([])`; the dashboard Clear does the same |
| Reply budget | **us** | `LLM_MAX_TOKENS = 300` |
| Context size logged | **us** | Per turn: `context_tokens`, `context_messages` |

## What is not managed

Only one thing, and it is the message list itself: **no trimming, no
summarisation, no prompt cache.** Nothing else.

That is a switch, not a build. Pipecat 1.15 already ships
`LLMContextSummarizer`, instantiated for every universal aggregator
(`llm_response_universal.py:1870`), with `LLMAutoContextSummarizationConfig`
defaulting to `max_context_tokens=8000` / `max_unsummarized_messages=20`. It
does nothing because `enable_auto_context_summarization` defaults to `False`
(`:266`). Turning it on is a config flag plus a summary model.

## Measured, over the run

| | Value |
|---|---|
| LLM calls | 94 |
| Conversation tokens per call | median 879, max 1,674 |
| Messages in context | median 47.5, max 94 |
| First → last call | 7 → 1,674 tokens |
| **Full input per call** | **median 1,914, max 2,709** |

The 879 is the **conversation only**. `_context_size` sums `ctx.messages`, and
the prompt is not in the context — it rides on `system_instruction`, which the
adapter prepends at request time (`pipecat_marks.py:54-74`). The 4,140-char
prompt is another ~1,035 tokens, so the endpoint really sees about 1,914 at the median. The
7 → 1,674 curve still measures the conversation correctly; the row label was
what overstated its scope.

Per run: median 157, 514, 924 conversation tokens — the growth is the
conversation, not a leak.

## The three decisions that shaped this

**The prompt is not in the context.** Putting it in both sent it twice, and the
stale copy in the context shadowed the refreshed clock line. Pipecat warns
about exactly this and calls the context copy deprecated.

**The result cap is structural, and the ladder is what makes it safe.** A blind
slice at 1,200 chars turned a 5-result news payload into 2 usable results with
the `date` cut. `_cap_for_model` keeps whole results and reports how many were
dropped. The ladder then had a second bug: it walked `(whole, 700, …)` without
bisecting, so a result one character over the cap fell all the way to 700 — a
12,094-char `read_file` came back as 701 chars. Rungs at 4,000/3,000/2,000 fix
it; the same file now keeps 4,001. That is why the cap could come down from
24,000 to 8,000 without losing a document.

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
— starting fresh") would convert a silent hang into a recoverable turn. The
summarizer above is the trim; the spoken line is still missing.
