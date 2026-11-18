"""Do the tool instructions actually reach the model, and do they hold?

Three failures were reported against the live agent, and all three are about
the model's *beliefs* rather than about a broken tool:

1. TIME. "It fails to use the tool for telling the time." Measured on the live
   model with the real 14-tool set: the plain question is answered from the
   prompt clock without a tool call (correct), and a timezone question does
   call `bash TZ="..." date` (correct). The failure is the prompt clock going
   STALE -- a frozen clock is quoted confidently, because nothing in the prompt
   distinguishes "given" from "current":

       [clock frozen at 06:30, real time 11:33]
       "What time is it?"  ->  no call, "It is 6:30 AM."   WRONG by 5 hours

   The guard says "never state a clock time you have not been given", and a
   stale clock *was* given, so the guard does not fire. The fix is that the
   clock must be re-read, which is what this test pins.

2. GITHUB. "It claims it does not have access even though the token was
   provided." The token was fine the whole time (github_repo action=list
   returned real repos in 1.8s). The model was inferring its capability from
   the tool description, which says "Requires GITHUB_TOKEN":

       "Can you push to my GitHub?"    -> "I can't push directly yet"  FALSE
       "Do you have my GitHub token?"  -> "I do not have access"       FALSE
       "Can you see my repos?"         -> "I sure can!" (no call)      UNCHECKED

   Fixed by naming GitHub in TOOLS_SYSTEM_PROMPT and requiring a call before
   any claim, plus a confirmation before any write.

3. LONG TASKS. `bash` is foreground; a measured 60s command held the call for
   the full 60s, and MCP_TOOL_TIMEOUT is 320s. `serve` returns in 0.5s. The
   prompt said nothing about which to use, so "run ./build.sh" went to bash.
   Fixed by routing anything slow to `serve`.

Run: .venv/bin/python test_tool_guidance.py
"""
from __future__ import annotations

import re
import sys

import config

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  {mark}  {label}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    if not ok:
        FAILS.append(label)
    return ok


# ---------------------------------------------------------------------------
# 1. The GitHub instruction
# ---------------------------------------------------------------------------
def test_github_instruction() -> bool:
    print("\n1. the prompt tells the model it has GitHub access")
    p = config.TOOLS_SYSTEM_PROMPT
    ok = check("it names the github_repo tool", "github_repo" in p)
    ok &= check("it says the token is configured", "token is configured" in p)
    ok &= check("it forbids claiming no access without checking",
                "Never claim you cannot reach GitHub" in p)
    ok &= check("it forbids claiming access without checking",
                "never claim you can, without checking first" in p)
    ok &= check("it requires a read to confirm", "action list" in p)
    ok &= check("it requires confirmation before a write",
                "wait for the person to confirm" in p)
    ok &= check("it separates a missing scope from no access",
                "do not describe a missing permission as not having access" in p)
    return ok


# ---------------------------------------------------------------------------
# 2. The long-task instruction
# ---------------------------------------------------------------------------
def test_long_task_instruction() -> bool:
    print("\n2. the prompt routes slow work to the detached tool")
    p = config.TOOLS_SYSTEM_PROMPT
    ok = check("it names the serve tool", "`serve`" in p)
    ok &= check("it says bash is foreground", "foreground" in p)
    ok &= check("it lists the slow cases", "build" in p and "install" in p)
    ok &= check("it forbids claiming done before serve says so",
                "Do not claim a long task is done" in p)
    ok &= check("it says how to read progress", "action logs" in p)
    return ok


# ---------------------------------------------------------------------------
# 3. The clock is unambiguous
# ---------------------------------------------------------------------------
def test_clock_unambiguous() -> bool:
    print("\n3. the clock line cannot be read two ways")
    line = config.clock_line()
    print(f"     {line.strip()}")
    ok = check("it carries an IANA zone name",
               bool(re.search(r"\(\w+/\w+", line)))
    # PST is Philippine Standard Time here and Pacific Standard Time almost
    # everywhere else. The offset must not disagree with the abbreviation.
    ok &= check("PST is not paired with a negative offset",
                not re.search(r"\bPST\b, UTC-", line))
    ok &= check("the offset matches the host", "UTC+0800" in line)
    return ok


# ---------------------------------------------------------------------------
# 4. A stale clock is actually replaced
# ---------------------------------------------------------------------------
def test_stale_clock_is_replaced() -> bool:
    print("\n4. a clock frozen hours ago is refreshed before inference")
    from agent import ClockedLLMService

    stale_time = "06:30 AM"
    stale = ("You are WanDog. " + config.OWNER_PROMPT + " "
             "The current date and time here is Wednesday, 18 November 2026, "
             + stale_time + " (Asia/Manila, PST, UTC+0800). "
             + config.CLOCK_GUARD + " " + config.TOOLS_SYSTEM_PROMPT).strip()

    class S:
        system_instruction = stale

    obj = object.__new__(ClockedLLMService)
    obj._settings = S()
    obj._base_system_instruction = stale
    obj._refresh_clock()
    after = obj._settings.system_instruction

    ok = check("the stale time is gone", stale_time not in after)
    ok &= check("exactly one clock remains",
                len(re.findall(r"The current date and time here is", after)) == 1)
    ok &= check("the prompt did not grow", len(after) == len(stale),
                f"{len(stale)} -> {len(after)}")
    ok &= check("the owner line survived", config.OWNER_PROMPT in after)
    return ok


# ---------------------------------------------------------------------------
# 5. The guard and the clock do not contradict each other
# ---------------------------------------------------------------------------
def test_guard_and_clock_agree() -> bool:
    print("\n5. the guard, the clock and the timezone hint agree")
    ok = check("the guard forbids invented times",
               "Never state a clock time you have not been given" in config.CLOCK_GUARD)
    ok &= check("the guard is in the tools prompt",
                config.CLOCK_GUARD in config.TOOLS_SYSTEM_PROMPT)
    ok &= check("the timezone recipe uses date", "TZ=" in config.TOOLS_SYSTEM_PROMPT)
    ok &= check("the clock satisfies the guard (it is 'given')",
                "The current date and time here is" in config.clock_line())
    return ok


def main() -> int:
    results = [
        test_github_instruction(),
        test_long_task_instruction(),
        test_clock_unambiguous(),
        test_stale_clock_is_replaced(),
        test_guard_and_clock_agree(),
    ]
    print("\n--- summary ---")
    print(f"  {sum(results)}/{len(results)} groups passed")
    if FAILS:
        print("\nFAILURES ABOVE")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
