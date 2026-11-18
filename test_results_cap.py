"""Is the model's result cap hard, and does the cap cost the agent anything?

`config.MCP_MODEL_RESULT_CHARS` (1200) bounds how much of a tool result the LLM
sees. The cap itself is deliberate -- a voice reply is one or two sentences --
so this test does not ask for it to be removed. It asks whether the cap is
*hard* (does any payload get past it?) and whether being hard costs the agent
an answer it should have had.

Two defects were found and fixed, both of them the same shape as the earlier
empty-result bug: the cap produced output the model could not use.

1. DICT PAYLOADS WERE CUT MID-STRING. `_cap_json_list` only handled a JSON
   *list*, which is what `search` returns. Every other tool returns a dict --
   `local_search`, `grep`, `read_file`, `search_document`, `get_weather` -- and
   a dict fell through to the blind `text[:limit]` slice. Measured on the four
   shapes the model can receive, all four came back as invalid JSON:

       local_search     1718 -> 1240 chars, INVALID
       read_file       12094 -> 1241 chars, INVALID
       grep             3990 -> 1240 chars, INVALID
       search_document  2828 -> 1240 chars, INVALID

   It stayed hidden because the event log contains only `search`,
   `get_weather` and `bash` calls -- and of those, only `search` ever exceeded
   the cap (35 of 36 searches; every `bash` result was 140-172 chars). The
   tools that break had not been used yet. Fixed by `_cap_json_obj`.

2. THE LAST RESORT STILL RETURNED AN EMPTY LIST. When one result's identity
   fields alone exceeded the budget, `_cap_json_list` answered
   `{"results": [], "showing": 0}`. That is the exact failure the two-stage cap
   was written to remove, just gated behind a 1200+ char url. Measured: a
   2000-char url produced "no result could be shown" for a payload holding one
   real result. Fixed by clipping the identity fields and keeping the result.

Offline: no MCP server, no key, no mic.

Run: .venv/bin/python test_results_cap.py
"""
from __future__ import annotations

import json

import config
import mcp_bridge

CAP = config.MCP_MODEL_RESULT_CHARS


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def _valid(out: str) -> bool:
    try:
        json.loads(out)
        return True
    except (ValueError, TypeError):
        return False


def _shown(out: str) -> int:
    """How many records survived the cap, or -1 if the output is unparseable."""
    try:
        d = json.loads(out)
    except (ValueError, TypeError):
        return -1
    if isinstance(d, dict):
        if "showing" in d:
            return d["showing"]
        for key in ("results", "sections", "documents"):
            if isinstance(d.get(key), list):
                return len(d[key])
        return 0
    return len(d)


# --- the payload shapes the real tools return -------------------------------

def local_search_payload(n: int = 5) -> str:
    return json.dumps({
        "query": "result cap", "method": "hybrid",
        "results": [{"rank": i, "score": 9.5 - i, "file": "WanDog-Notes.md",
                     "location": f"lines {i * 10}-{i * 10 + 8}",
                     "text": "some matched chunk text about the news cap " * 4}
                    for i in range(1, n + 1)],
        "documents": [{"file": "WanDog-Notes.md", "best_rank": 1,
                       "matched_at": "lines 150-160",
                       "opening": "The latest-news fixes: does a search result "
                                  "reach the model intact and dated? Three "
                                  "defects were measured on the live agent.",
                       "num_chunks": 12}],
        "total_results": n, "total_documents": 1, "total_chunks": 40,
        "status": "success",
    }, ensure_ascii=False)


def grep_payload(n: int = 60) -> str:
    return json.dumps({
        "results": [{"file": "a.py", "line_number": i,
                     "content": f"match line {i}"} for i in range(n)],
        "total_matches": n, "pattern": "x", "directory": ".",
        "file_pattern": "*", "status": "success",
    })


def read_file_payload(size: int = 12000) -> str:
    return json.dumps({"content": "x" * size, "path": "/a/b.py",
                       "size_bytes": size, "format": "text",
                       "status": "success"})


def search_document_payload(n: int = 6) -> str:
    return json.dumps({
        "sections": [{"text": "passage about the cap " * 20} for _ in range(n)],
        "total_sections": n, "query": "cap", "file": "a.md", "format": "md",
        "status": "success",
    })


def web_payload(n: int = 5, body: int = 1400) -> str:
    return json.dumps([{
        "title": f"Result {i} about CES 2025",
        "url": f"https://example.com/page-{i}",
        "content": ("CES is the most powerful tech event in the world. " * 40)[:body],
    } for i in range(n)])


# --- 1. the cap is hard -----------------------------------------------------

def test_cap_is_hard() -> bool:
    """No payload may exceed the budget by more than the trailer's own length."""
    print("\n[1] the cap is hard -- nothing sneaks past it")
    ok = True
    # The structural paths stay under the limit exactly.
    for label, raw, q in [
        ("local_search", local_search_payload(), "result cap"),
        ("grep", grep_payload(), "match"),
        ("read_file", read_file_payload(), None),
        ("search_document", search_document_payload(), "cap"),
        ("web search", web_payload(), "when is CES 2025"),
    ]:
        out = mcp_bridge._cap_for_model(raw, CAP, query=q)
        ok &= check(f"{label}: {len(raw)} -> {len(out)} chars under {CAP}",
                    len(out) <= CAP, f"over by {len(out) - CAP}" if len(out) > CAP else "")
    # The plain-text path adds a trailer after the slice, so it is the one
    # shape that can read slightly over. Bound that explicitly rather than
    # pretending it fits.
    text = "x" * (CAP + 9000)
    out = mcp_bridge._cap_for_model(text, CAP)
    ok &= check(f"plain text: {len(text)} -> {len(out)} chars (trailer allowed)",
                len(out) <= CAP + 80, f"over by {len(out) - CAP}")
    ok &= check("plain text says how much was cut", "truncated" in out)
    return ok


# --- 2. dict payloads stay parseable ---------------------------------------

def test_dict_payloads_stay_valid() -> bool:
    """Every dict-shaped tool result must reach the model as valid JSON."""
    print("\n[2] dict payloads reach the model as parseable JSON")
    ok = True
    cases = [
        ("local_search", local_search_payload(), "result cap"),
        ("grep", grep_payload(), "match"),
        ("read_file", read_file_payload(), None),
        ("search_document", search_document_payload(), "cap"),
        ("get_weather (big)", json.dumps({
            "location": "Tokyo", "current": {"description": "rain",
                                             "temperature_c": 11.7},
            "forecast_5day": [{"d": i, "t": 20 + i} for i in range(5)],
            "junk": "x" * 3000}), None),
    ]
    for label, raw, q in cases:
        out = mcp_bridge._cap_for_model(raw, CAP, query=q)
        ok &= check(f"{label} parses", _valid(out),
                    "" if _valid(out) else f"tail={out[-40:]!r}")
    return ok


def test_dict_cap_was_broken_before() -> bool:
    """Pin the old behaviour, so a regression is unambiguous."""
    print("\n[3] the old blind slice produced invalid JSON (regression pin)")
    ok = True
    raw = grep_payload()
    # What the code did before _cap_json_obj: a raw prefix slice.
    blind = raw[:CAP].rstrip() + f"\n[truncated: showing {CAP} of {len(raw)} chars]"
    ok &= check("a blind slice of a dict payload does NOT parse", not _valid(blind))
    ok &= check("the cap no longer does that", _valid(mcp_bridge._cap_for_model(raw, CAP)))
    return ok


def test_dict_keeps_identity_and_counts() -> bool:
    """Trimming records must not lose the url, the date, or the true totals."""
    print("\n[4] dict cap keeps identity fields and true totals")
    ok = True
    raw = local_search_payload()
    d = json.loads(mcp_bridge._cap_for_model(raw, CAP, query="result cap"))
    ok &= check("query survives", d.get("query") == "result cap")
    ok &= check("total_results is the server's count, not the shown count",
                d.get("total_results") == 5, str(d.get("total_results")))
    ok &= check("status survives", d.get("status") == "success")
    ok &= check("documents survive", bool(isinstance(d.get("documents"), list)
                and d["documents"]))
    ok &= check("results survive", bool(isinstance(d.get("results"), list)
                and d["results"]))

    # The default budget (24000) is larger than this 1718-char payload, so
    # nothing is trimmed and no note is owed. Pin that: a payload that fits
    # must be passed through untouched, note and all.
    ok &= check("a payload under the default budget is untouched",
                mcp_bridge._cap_for_model(raw, CAP) == raw)
    # Force the trimming path with an explicit small limit, which is where the
    # note is owed. The limit is passed rather than taken from config so this
    # keeps testing the trimming code after the default was raised.
    d = json.loads(mcp_bridge._cap_for_model(raw, 900, query="result cap"))
    ok &= check("a note explains any trimming (at a limit that trims)",
                bool(d.get("note")), str(d.get("note"))[:60])

    raw = json.dumps([{"title": "T", "url": "https://e.com/1",
                       "date": "November 17, 2026", "content": "b" * 2000}])
    d = json.loads(mcp_bridge._cap_for_model(raw, CAP))
    item = d["results"][0] if isinstance(d, dict) else d[0]
    ok &= check("url intact", item.get("url") == "https://e.com/1")
    ok &= check("date intact", item.get("date") == "November 17, 2026")
    return ok


# --- 3. the last resort never returns an empty list -------------------------

def test_never_empty() -> bool:
    """No input, however shaped, may produce an empty result list."""
    print("\n[5] no payload ends in an empty result list")
    ok = True
    cases = {
        "url 2000 chars": json.dumps([{"title": "T",
                                       "url": "https://e.com/" + "a" * 2000,
                                       "content": "body"}]),
        "url 5000 chars": json.dumps([{"title": "T",
                                       "url": "https://e.com/" + "a" * 5000,
                                       "content": "body"}]),
        "title 1500 chars": json.dumps([{"title": "T" * 1500,
                                         "url": "https://e.com/x",
                                         "content": "b"}]),
        "five results, huge urls": json.dumps([
            {"title": "T" * 300, "url": "https://e.com/" + "a" * 900,
             "content": "b" * 900} for _ in range(5)]),
    }
    for label, raw in cases.items():
        out = mcp_bridge._cap_for_model(raw, CAP)
        ok &= check(f"{label}: parses", _valid(out))
        ok &= check(f"{label}: still shows a result", _shown(out) >= 1,
                    f"showing={_shown(out)}")
    return ok


def test_empty_result_was_reachable_before() -> bool:
    """The old code's empty answer, reproduced, so the fix is not theoretical."""
    print("\n[6] the empty result WAS reachable (regression pin)")
    raw = json.dumps([{"title": "T", "url": "https://e.com/" + "a" * 2000,
                       "content": "body"}])
    old = json.dumps({"results": [], "showing": 0, "total": 1,
                      "note": "the first result alone is larger than 1200 chars"})
    ok = check("the old answer held zero results", _shown(old) == 0)
    ok &= check("the current answer holds one", _shown(
        mcp_bridge._cap_for_model(raw, CAP)) == 1)
    return ok


# --- 4. the cap costs sources, and that is measured -------------------------

def test_cap_costs_sources() -> bool:
    """What the budget buys, at the default and at the old 1200.

    The default is now 24000, and the point of raising it was that *no observed
    payload* is truncated at that size -- a web page with full bodies and a
    5-result news payload both arrive whole. The 1200 column is kept as the
    measured reason for the change: it showed 2 of 5 sources, and on a real
    news payload it showed 2 of 5 where 5 fit at 24000.
    """
    print("\n[7] what the budget buys (measured, not asserted as correct)")
    ok = True

    # At the default, the shapes the tools really return are not cut at all.
    for label, raw, q in [
        ("5-result web page, full bodies", web_payload(), "when is CES 2025"),
        ("local_search", local_search_payload(), "result cap"),
        ("grep (60 matches)", grep_payload(), "match"),
        ("search_document", search_document_payload(), "cap"),
    ]:
        out = mcp_bridge._cap_for_model(raw, CAP, query=q)
        ok &= check(f"{label}: {len(raw)} chars fits whole at {CAP}",
                    out == raw, f"got {len(out)} chars")

    # A result bigger than the default is still cut, and must stay parseable
    # and keep its path -- that is what the dict cap is for. read_file's own
    # 12,094-char default payload now fits whole, so ask for a bigger file.
    big = read_file_payload(30000)
    out = mcp_bridge._cap_for_model(big, CAP)
    d = json.loads(out)
    ok &= check(f"read_file {len(big)} chars is cut to {len(out)} at {CAP}",
                len(out) <= CAP and _valid(out), f"over by {len(out) - CAP}")
    ok &= check("read_file keeps its path and size",
                d.get("path") == "/a/b.py" and d.get("size_bytes") == 30000,
                f"path={d.get('path')!r} size={d.get('size_bytes')!r}")

    # The old budget, for the record.
    out = mcp_bridge._cap_for_model(web_payload(), 1200, query="when is CES 2025")
    ok &= check("at 1200 the same web page kept only 2 of 5 sources",
                _shown(out) == 2, f"showing={_shown(out)}")
    news = open("/tmp/news_payload.json", encoding="utf-8").read() \
        if __import__("os").path.exists("/tmp/news_payload.json") else None
    if news:
        sorted_news = mcp_bridge._sort_news(news, 5)
        small = mcp_bridge._cap_for_model(sorted_news, 1200, query="latest news")
        ok &= check("a real 5-result news payload is cut at 1200 chars",
                    _shown(small) < 5, f"showing={_shown(small)} of 5")
        out = mcp_bridge._cap_for_model(sorted_news, CAP, query="latest news")
        ok &= check(f"and fits whole at the {CAP} default",
                    out == sorted_news, f"{len(out)} vs {len(sorted_news)} chars")
    return ok


def test_default_is_large_enough() -> bool:
    """The default must leave a payload usable, not merely cut it.

    The old form of this test asserted CAP >= 20000, on the reasoning that a
    cap under the largest observed payload (22,970 chars) is a cap that decides
    the answer. That reasoning was sound only while _CAP_CLIP_LADDER had a
    single rung at 700: a result one char over the cap fell all the way to 701
    chars, so anything under the largest payload really was destroyed. With the
    4000/3000/2000 rungs the cap degrades instead of collapsing, and what
    matters is no longer "does every payload fit whole" but "does an oversized
    one keep enough to answer from".
    """
    print("\\n[8] the default leaves an oversized payload usable")
    ok = check(f"MCP_MODEL_RESULT_CHARS = {CAP}", CAP >= 8000,
               "must clear the 6,280-char largest result in the event log")
    # A payload just over the default is still cut, and still says so: the cap
    # did not become decorative.
    raw = json.dumps([{"title": "T", "url": "https://e.com/1",
                       "date": "November 17, 2026",
                       "content": "b" * (CAP + 4000)}])
    out = mcp_bridge._cap_for_model(raw, CAP)
    ok &= check(f"a {len(raw)}-char result is still cut to {len(out)}",
                len(out) <= CAP and out != raw)
    # The point of the ladder: an oversized record keeps thousands of chars of
    # prose, not hundreds. Before the rungs this was 700 for any size.
    kept = len(json.loads(out)["results"][0]["content"])
    ok &= check(f"the oversized record still keeps {kept} chars of prose",
                kept >= 2000, "the ladder must not fall to the 700 rung")
    ok &= check("the record's url and date survived",
                bool(json.loads(out)["results"][0]["url"]
                     and json.loads(out)["results"][0]["date"]))
    return ok


def test_zero_disables() -> bool:
    """limit <= 0 must still mean "no cap" -- the escape hatch stays open."""
    print("\n[9] limit 0 disables the cap")
    ok = True
    raw = grep_payload()
    ok &= check("0 returns the payload untouched",
                mcp_bridge._cap_for_model(raw, 0) == raw)
    ok &= check("a small payload is untouched too",
                mcp_bridge._cap_for_model('{"ok": true}', CAP) == '{"ok": true}')
    return ok


def main() -> int:
    results = [
        test_cap_is_hard(),
        test_dict_payloads_stay_valid(),
        test_dict_cap_was_broken_before(),
        test_dict_keeps_identity_and_counts(),
        test_never_empty(),
        test_empty_result_was_reachable_before(),
        test_cap_costs_sources(),
        test_default_is_large_enough(),
        test_zero_disables(),
    ]
    print("\n--- summary ---")
    print(f"  {sum(results)}/{len(results)} groups passed")
    print("\nALL PASSED" if all(results) else "\nFAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
