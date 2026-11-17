"""Offline tests for news-v2: freshness of news search results.

The bug this covers: on 17 Nov 2026 the agent answered "the latest news is
November 12" for the Alaska Senate race while Google showed stories from that
morning. Nothing was broken in the transport -- the model was handed a
correct, dated, well-formed payload whose first two entries were days old.
ddgs orders news by relevance, not recency, and the model cap keeps a prefix,
so the prefix decided the answer.

Run:  .venv/bin/python test_news_v2.py
"""

from __future__ import annotations

import datetime
import json

import config
import mcp_bridge


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def _payload(*dates: str) -> str:
    return json.dumps([
        {"title": f"story {i}", "snippet": "body", "date": d,
         "source": "APM", "url": f"https://example.com/{i}"}
        for i, d in enumerate(dates)
    ])


def _today() -> str:
    d = datetime.date.today()
    return d.strftime("%B %d, %Y").replace(" 0", " ")


def _days_ago(n: int) -> str:
    d = datetime.date.today() - datetime.timedelta(days=n)
    return d.strftime("%B %d, %Y").replace(" 0", " ")


def test_date_parsing() -> bool:
    """Both shapes OnIt and ddgs emit must parse."""
    print("\n[1] date parsing")
    ok = True
    iso = mcp_bridge._parse_news_date("2026-11-17T03:56:25+00:00")
    ok &= check("ISO stamp parses", iso is not None and iso.day == 17)
    human = mcp_bridge._parse_news_date("November 17, 2026")
    ok &= check("OnIt's '%B %d, %Y' parses", human is not None and human.month == 11)
    ok &= check("garbage is None, not an exception",
                mcp_bridge._parse_news_date("last tuesday") is None)
    ok &= check("empty is None", mcp_bridge._parse_news_date("") is None)
    ok &= check("non-string is None", mcp_bridge._parse_news_date(None) is None)
    return ok


def test_sort_newest_first() -> bool:
    """The Alaska shape: relevance order 17, 12, 17, 14, 11 -> 17 first."""
    print("\n[2] newest-first sort")
    ok = True
    raw = _payload(_days_ago(5), _days_ago(0), _days_ago(3), _days_ago(6), _days_ago(0))
    out = json.loads(mcp_bridge._sort_news(raw, 5))
    ok &= check("newest item is first", out[0]["date"] == _days_ago(0), out[0]["date"])
    ok &= check("second is also today", out[1]["date"] == _days_ago(0))
    dates = [mcp_bridge._parse_news_date(r["date"]) for r in out]
    ok &= check("strictly non-increasing", all(a >= b for a, b in zip(dates, dates[1:])))
    ok &= check("nothing lost (5 in, 5 out)", len(out) == 5, f"{len(out)}")
    return ok


def test_age_labels() -> bool:
    """`age` is what makes the model say 'today' instead of reading a date."""
    print("\n[3] age labels")
    ok = True
    out = json.loads(mcp_bridge._sort_news(
        _payload(_days_ago(0), _days_ago(1), _days_ago(4)), 3))
    ok &= check("today labelled 'today'", out[0].get("age") == "today", out[0].get("age"))
    ok &= check("yesterday labelled 'yesterday'", out[1].get("age") == "yesterday",
                out[1].get("age"))
    ok &= check("4 days labelled '4 days ago'", out[2].get("age") == "4 days ago",
                out[2].get("age"))
    ok &= check("original fields survive",
                bool(out[0].get("url") and out[0].get("source") and out[0].get("title")))
    return ok


def test_trim_to_wanted() -> bool:
    """Over-fetching must not leak extra results into the model's view."""
    print("\n[4] over-fetch then trim")
    ok = True
    raw = _payload(*[_days_ago(i) for i in range(10)])
    out = json.loads(mcp_bridge._sort_news(raw, 5))
    ok &= check("asked 5, get 5", len(out) == 5, f"{len(out)}")
    ok &= check("trimmed from the OLD end", out[-1]["date"] == _days_ago(4), out[-1]["date"])
    full = json.loads(mcp_bridge._sort_news(raw, None))
    ok &= check("wanted=None keeps everything", len(full) == 10, f"{len(full)}")
    return ok


def test_shape_overfetches() -> bool:
    """_shape_search_args must ask the server for more than the caller wants."""
    print("\n[5] over-fetch in _shape_search_args")
    ok = True
    shaped = mcp_bridge._shape_search_args(
        {"query": "latest news on the alaska senate race", "type": "web", "max_results": 5})
    ok &= check("type becomes news", shaped.get("type") == "news")
    ok &= check("asks for MCP_NEWS_OVERFETCH", shaped.get("max_results") == config.MCP_NEWS_OVERFETCH,
                str(shaped.get("max_results")))
    ok &= check("remembers the caller's 5", shaped.get("_news_wanted") == 5,
                str(shaped.get("_news_wanted")))
    small = mcp_bridge._shape_search_args(
        {"query": "latest news", "type": "web", "max_results": 2})
    ok &= check("a 2-result request still over-fetches",
                small.get("max_results") >= config.MCP_NEWS_OVERFETCH)
    ok &= check("and remembers 5, not 2", small.get("_news_wanted") == 5,
                str(small.get("_news_wanted")))
    return ok


def test_private_key_never_sent() -> bool:
    """`_news_wanted` is ours; the MCP server would reject it."""
    print("\n[6] private key stays local")
    ok = True
    shaped = mcp_bridge._shape_search_args({"query": "latest news", "type": "web"})
    stripped = {k: v for k, v in shaped.items() if not str(k).startswith("_")}
    ok &= check("strip removes _news_wanted", "_news_wanted" not in stripped)
    ok &= check("strip keeps query/type/max_results",
                {"query", "type", "max_results"} <= set(stripped), str(sorted(stripped)))
    return ok


def test_newest_age() -> bool:
    """The refresh decision reads the newest date in the payload."""
    print("\n[7] newest-age helper")
    ok = True
    ok &= check("fresh page -> 0", mcp_bridge._news_newest_age_days(_payload(_days_ago(0))) == 0)
    ok &= check("5-day page -> 5", mcp_bridge._news_newest_age_days(_payload(_days_ago(5))) == 5)
    mixed = _payload(_days_ago(5), _days_ago(0))
    ok &= check("mixed page reports the NEWEST", mcp_bridge._news_newest_age_days(mixed) == 0)
    ok &= check("no dates -> None", mcp_bridge._news_newest_age_days('[{"title":"x"}]') is None)
    ok &= check("error object -> None",
                mcp_bridge._news_newest_age_days('{"error":"Search failed"}') is None)
    ok &= check("non-JSON -> None", mcp_bridge._news_newest_age_days("hello") is None)
    return ok


def test_untouched_shapes() -> bool:
    """Sorting must be a no-op on anything that is not a news list."""
    print("\n[8] non-news payloads pass through")
    ok = True
    for name, payload in [
        ("error object", '{"error": "Search failed"}'),
        ("web results (no date)", '[{"title":"x","url":"u","content":"c"}]'),
        ("plain text", "total 0\ndrwxr-xr-x 2 user user"),
        ("empty list", "[]"),
        ("list of strings", '["a","b"]'),
    ]:
        ok &= check(f"{name} unchanged", mcp_bridge._sort_news(payload, 5) == payload)
    return ok


def test_switch_off() -> bool:
    """MCP_NEWS_SORT=0 must restore the old pass-through exactly."""
    print("\n[9] sort can be switched off")
    ok = True
    raw = _payload(_days_ago(5), _days_ago(0))
    original = config.MCP_NEWS_SORT
    try:
        config.MCP_NEWS_SORT = False
        ok &= check("off -> payload untouched", mcp_bridge._sort_news(raw, 5) == raw)
    finally:
        config.MCP_NEWS_SORT = original
    ok &= check("on again -> sorted", json.loads(mcp_bridge._sort_news(raw, 5))[0]["date"] == _days_ago(0))
    return ok


def test_cap_sees_the_fresh_one() -> bool:
    """The whole point: after sort+cap, the model's FIRST result is the newest."""
    print("\n[10] the cap keeps the fresh result")
    ok = True
    raw = _payload(_days_ago(5), _days_ago(0), _days_ago(3), _days_ago(6), _days_ago(0))
    # Old behaviour: cap the server order directly.
    old = json.loads(mcp_bridge._cap_for_model(raw, config.MCP_MODEL_RESULT_CHARS,
                                              query="alaska senate results"))
    old_rows = old["results"] if isinstance(old, dict) else old
    # New behaviour: sort, then cap.
    new = json.loads(mcp_bridge._cap_for_model(mcp_bridge._sort_news(raw, 5),
                                              config.MCP_MODEL_RESULT_CHARS,
                                              query="alaska senate results"))
    new_rows = new["results"] if isinstance(new, dict) else new
    ok &= check("old first result was NOT today", old_rows[0]["date"] != _days_ago(0),
                old_rows[0]["date"])
    ok &= check("new first result IS today", new_rows[0]["date"] == _days_ago(0),
                new_rows[0]["date"])
    ok &= check("new payload carries age", new_rows[0].get("age") == "today")
    return ok


def test_defaults_sane() -> bool:
    """Config values that the voice latency depends on."""
    print("\n[11] config defaults")
    ok = True
    ok &= check("over-fetch is 10 (ddgs' practical ceiling)",
                config.MCP_NEWS_OVERFETCH == 10, str(config.MCP_NEWS_OVERFETCH))
    ok &= check("refresh retries bounded (<=3)", 0 <= config.MCP_NEWS_REFRESH_RETRIES <= 3,
                str(config.MCP_NEWS_REFRESH_RETRIES))
    ok &= check("refresh budget under 5s", 0 < config.MCP_NEWS_REFRESH_BUDGET <= 5,
                str(config.MCP_NEWS_REFRESH_BUDGET))
    ok &= check("fresh window is 0 days (today only)", config.MCP_NEWS_FRESH_DAYS == 0,
                str(config.MCP_NEWS_FRESH_DAYS))
    ok &= check("sort on by default", config.MCP_NEWS_SORT is True)
    return ok


def main() -> int:
    results = [
        test_date_parsing(),
        test_sort_newest_first(),
        test_age_labels(),
        test_trim_to_wanted(),
        test_shape_overfetches(),
        test_private_key_never_sent(),
        test_newest_age(),
        test_untouched_shapes(),
        test_switch_off(),
        test_cap_sees_the_fresh_one(),
        test_defaults_sane(),
    ]
    print("\n--- summary ---")
    print(f"  {sum(results)}/{len(results)} groups passed")
    print("\nALL PASSED" if all(results) else "\nFAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
