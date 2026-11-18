# WanDog — Next Steps, branch `sidebar`

**Date:** November 18, 2026
**Branch:** `sidebar` at `ec8ec091`, created from `file-upload` (`8ca5e8e5`)
**Rollback:** `git switch main` — `main` stays at `392cc979` and is never committed to.
**Status:** this document is a plan and a verified record. Every number and every
`file:line` below was checked against the tree on this branch, today. Where a check
failed, the correction is written into the document rather than quietly fixed.

**Read this first:** §1 is the only urgent section. Three of the four open faults are
recorded as `PENDING` in `WanDog-Notes.md` while their fixes are already in this
branch's history, which means the notes currently contradict the code.

---

## 0. Where this branch stands

Five commits ahead of `main`, 18 files, +4 303 / −172.

| Commit   | What it did                                                 |
|----------|-------------------------------------------------------------|
| 97956495 | sessions, the per-session tool jail, the upload question     |
| c2cfb498 | notes §23 and both registers                                 |
| a9ddfe2a | `test_upload`: the dashboard imports `session` as `sess`      |
| 8ca5e8e5 | `test_session_launch`: use a throwaway sessions root         |
| ec8ec091 | the sidebar: listed, browsed, switchable, with handover      |

**The chain matters for the merge.** `sidebar` sits on `file-upload`, which sits on
`main`. Merging `sidebar` lands all three features at once; landing `file-upload`
first and `sidebar` after keeps the two decisions separate. Both routes respect the
rule that `main` is never committed to directly. §6 records this as a decision to
make, not a task to do.

### Test baseline, run today on this branch

35 test files, no pytest in the venv, so every file is run directly with
`.venv/bin/python <file>`. Verified today:

| Result  | Files                                                                    |
|---------|--------------------------------------------------------------------------|
| pass    | `test_session`, `test_session_jail`, `test_session_launch`,               |
|         | `test_session_replay`, `test_sidebar`, `test_upload`, `test_upload_e2e`,  |
|         | `test_eventbus`, `test_results_cap`, `test_clock`, `test_tool_parity`,    |
|         | `test_tool_result`, `test_turn_context`, `test_guard`, `test_dashboard_ui`,|
|         | `test_dashboard_controls`, `test_dashboard_layout`                        |
| fail    | `test_ctrl_enter_interrupt` — the composer reports `none` both idle and in |
|         | flight, so the busy case is never exercised                                |
| fail    | `test_latest_news` — 12/13 groups; a live Tavily call returned nothing     |

The two failures are not caused by this branch. Neither file is touched by the five
commits (`git diff --stat main..sidebar` lists neither), and both fail identically on
`main`.

---

## 1. URGENT — the notes contradict the code

This is first because on this project the notes *are* the handover. A future session
reads them instead of reading the diff.

`WanDog-Notes.md:1639-1642` records four faults as `PENDING`. All four were fixed in
`ec8ec091` — the same commit that wrote the rows:

| Row says | Where the fix actually is                                        |
|----------|-------------------------------------------------------------------|
| Transcript 80 px short of the main area | `dashboard.py:324` `max-width: none`           |
| The bar ran 300 px past the window edge | `dashboard.py:400` and `:583` `width: auto`    |
| Scrollbar inboard: container padding, not gap | `dashboard.py:321` side padding zeroed    |
| Switch compared against a stale session id | `agent.py:404` `if sid == self._session_id`    |

Verified with `git grep -c`: `max-width: none` and `width: auto` occur 0 times on
`main` and 1 and 4 times on `sidebar`; `sid == self._session_id` is 0 on `main`, 1 on
`sidebar`. The guard test also exists: `test_dashboard_layout.py:580`
`test_live_layout_at_a_wide_window`, and it passes.

**Fix:** change the four rows from `PENDING` to `ec8ec091`. Nothing else in §1 is
needed — the work is done; only the record is wrong.

Two smaller stale claims in `NEXT-STEPS.md`, which this document supersedes but does
not delete:

- `:636` says the sidebar is "still not built". It is built, on this branch.
- `:765` says "`main` stays at `f4836c85`". `main` is at `392cc979`.
- The header at `:1-8` still says "plan only. No code was changed" — true of that
  file when written, misleading now that §5 is marked **DONE**.

---

## 2. Nothing bounds the sessions root

There is no delete, forget or prune path anywhere: `grep -rn 'rmtree\|def delete\|
def forget\|def prune' session.py agent.py dashboard.py` returns nothing. The rail can
list a session and switch to it, and there is no way to remove one.

Today `../wandog-sessions` is empty (`ls` → no entries), so this is not yet a problem
in practice. It becomes one the first time the agent is used for real, because
nothing ever removes a session and every launch adds one.

**What to build:** `session.remove(sid)` that refuses the live session (the pointer
names it), refuses a malformed id (reuse `SESSION_ID_RE`), and moves the folder aside
before deleting so a half-failure leaves something recoverable. Then a control on the
rail. A rail row that can delete must also ask — the confirmation is the point, not
the delete.

**Why before §4:** §4 makes the list cheap; this keeps the list short. Doing §4 first
optimises a set that still grows without bound.

---

## 3. Two agents collide, and nothing stops it

There is no single-instance guard: `grep -rn 'O_EXCL\|fcntl\|flock\|getpid' session.py
agent.py config.py` returns nothing relevant. `eventbus.py:73` holds the only
`threading.Lock` in the codebase, and it guards an in-process object, not a file.

This was survivable before switching existed. It is worse now, because the switch and
handover requests live at the sessions **root**, not inside a session folder:

- `session.py:354` `request_switch` and `:368` `request_handover` both write to
  `switch_path()` / `handover_path()` under `root()`.
- `session.py:383` `take_at` deletes the file *before* returning it, so exactly one
  process gets it — and the other agent's user is left clicking a row that does
  nothing, with no error anywhere.

Two agents also append to the same `.current` session and each take control files the
other was meant to get.

**What to build:** one lock file beside the pointer, created `O_EXCL` at startup, and
a refusal that names the other pid rather than a stack trace. The pid has to be
recoverable from the file so a crashed agent does not lock the user out permanently —
store pid + start time, and treat a pid that is gone as stale.

---

## 4. The rail re-reads every log, twice a second

Measured today on throwaway roots with correctly-shaped session ids
(`^\d{4}-\d{2}-\d{2}T\d{4}-[0-9a-f]{4}$`, `session.py:70`):

| sessions × turns | on disk | `list_sessions()` |
|------------------|---------|-------------------|
| 20 × 500         | 1.6 MB  | 34.5 ms           |
| 60 × 500         | 4.9 MB  | 100.3 ms          |
| 60 × 1500        | 14.7 MB | 321.0 ms          |

Roughly 22 ms per MB of log, and the cost is linear in total bytes, not in rows.

The cause is `session.py:307`: `list_sessions` calls `read_turns(d.name, limit=0)` for
every session, and `limit=0` means *no cap* (`session.py:473` — `turns[-limit:] if
limit else turns`). So every row costs a full JSON parse of that session's entire
`events.jsonl`, to produce two things: a 60-character title and a turn count.

The caller is `dashboard.py:1065`, the first line of `@st.fragment(run_every=2.0)`
(`dashboard.py:1063`). That is 2.5 full re-reads of the whole root per second, on
Streamlit's script thread, before anything is drawn. At 60 × 1500 that is a third of
every two-second tick spent in a blocking parse.

**What to build:** keep `title` and `turns` in `meta.json`, updated as turns land
(`append_turn` already writes the log and the notes, so it can write the count too),
so a row costs one small read. If a cache is preferred instead, key it on
`(mtime, size)` per `events.jsonl` and re-read only what changed — but the count in
`meta.json` is smaller and survives restarts without a cold pass.

**One trap:** `list_sessions` currently tolerates a folder with no `meta.json`
(`session.py:303-305`) so a half-created session still lists. Whatever is stored in
`meta.json` has to keep that tolerance, or a session that is being created vanishes
from the rail at the moment it is most interesting.

---

## 5. There is no test runner

35 files, no `pytest` installed (`.venv/bin/python -c "import pytest"` → not found),
no `pyproject.toml`, `pytest.ini`, `setup.cfg` or `tox.ini`.

Two incompatible conventions coexist:

- 28 files have a `__main__` block, so they run as scripts. Only 11 of those
  actually define `def test_*` functions.
- 7 files use a local `check()` helper and have **no** `def test_` and no
  `__main__`: `test_session`, `test_session_jail`, `test_session_launch`,
  `test_session_replay`, `test_sidebar`, `test_upload`, `test_upload_e2e`. Between
  them: 292 `check()` calls, 0 `def test_` functions.

`test_sidebar.py` ends with `raise SystemExit(1)` on failure and a bare module body,
so it runs correctly as a script and would collect zero tests under pytest. The
consequence: **installing pytest would silently report success by collecting almost
nothing from the seven files that cover this branch's actual feature.**

**What to build:** `run_tests.py` that shells each file, times it, and prints
pass/fail/skip against a known-failing list (`test_ctrl_enter_interrupt`,
`test_latest_news`, and the network-dependent `*_live` files). That makes "N files
pass" reproducible by anyone instead of a claim in a transcript. If pytest is wanted
later, the seven `check()` files need `def test_*` wrappers first, and that is a
separate change.

---

## 6. The tier is visible but not settable

The dashboard shows the answering scope as a caption (`dashboard.py:1192-1201`), and
only when it is not the default — a narrowed scope is a limit the user must be able
to see. But there is no control to change it. `session.mode_from_text` is called from
exactly one place, `agent.py:1153`, inside the transcript path.

So `strict` is reachable only by answering the upload question, and widening back out
is reachable only by *saying* a phrase. Whether that is right is a product decision,
not a defect: for a voice agent, a spoken phrase is a coherent interface, and adding a
dropdown creates a second source of truth for the same state.

**Recommendation:** leave it, and note the decision. If a control is added later, it
must write the same `meta.json` field that `mode_from_text` writes, or the two paths
will disagree about what the agent is doing.

---

## 7. Order, and why

| # | Task                              | Cost      | Blocked by |
|---|-----------------------------------|-----------|------------|
| 1 | correct the four `PENDING` rows   | minutes   | nothing    |
| 2 | `session.remove` + a rail control | 1 commit  | nothing    |
| 3 | single-instance lock              | 1 commit  | nothing    |
| 4 | `title`/`turns` in `meta.json`    | 1 commit  | §2 (list stays short) |
| 5 | `run_tests.py`                    | 1 commit  | nothing    |
| 6 | tier control — decision, not work | —         | product call |

§1 first because it is the cheapest and the only one that misleads a reader. §2 before
§4 because a fast list of an unbounded set is still unbounded. §5 is independent and
can be done at any point; it is the one that makes every later claim checkable.

Nothing here requires undoing the sidebar work. §2 and §3 are additive; §4 changes
what `list_sessions` reads, not what it returns.

---

## 8. What this document does not do

- It does not change any code, and does not touch `main`, `file-upload` or `sidebar`.
- It does not fix the four `PENDING` rows. It says exactly what to change them to.
- It does not delete or supersede `NEXT-STEPS.md`. That file still documents tasks
  1-5, and §5 of it is the record of the upload build. The two stale claims it makes
  are listed in §1 above.
- It does not decide the merge route in §0. That is a decision for the branch owner.
- It does not run the network-dependent tests (`test_*_live`, `test_search_live`,
  `test_llm_live`), which need keys and a live endpoint.
