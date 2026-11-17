"""Offline tests for tool-secret resolution (no network, no API calls).

Covers the four properties that matter:

  1. secrets.yaml overrides the keychain.
  2. Deleting secrets.yaml falls back to the keychain -- the file is additive.
  3. A key present in neither is simply absent, and the agent still starts.
  4. YAML scalar retyping (int, bool) and blank values are handled.

Run:  .venv/bin/python test_tool_secrets.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(label)


def reset(monkeypatch: dict | None = None) -> None:
    """Drop the mtime cache and point at a given file, if any."""
    config._secrets_cache = None
    config._secrets_cache_mtime = None
    if monkeypatch is not None:
        config.TOOL_SECRETS_PATH = monkeypatch.get("path")


class FakeKeyring:
    """Stand-in for the OS keychain so the test needs no real secrets."""

    def __init__(self, values: dict[str, str]):
        self.values = values
        self.calls: list[str] = []

    def __call__(self, user: str):
        self.calls.append(user)
        return self.values.get(user)


def main() -> int:
    real_keyring = config._key_from_keyring
    tmpdir = tempfile.mkdtemp(prefix="fweddie-secrets-")
    secrets_file = Path(tmpdir) / "secrets.yaml"

    try:
        # -- 1. file overrides keychain -----------------------------------
        print("\n=== 1. secrets.yaml wins over the keychain ===")
        secrets_file.write_text(
            'openweathermap_api_key: "from-file"\n'
            'tavily_api_key: "tvly-file"\n'
        )
        config._key_from_keyring = FakeKeyring(
            {"openweathermap_api_key": "from-keychain",
             "tavily_api_key": "tvly-keychain",
             "github_token": "gh-keychain"}
        )
        reset({"path": secrets_file})
        check("file wins (weather)",
              config.resolve_tool_secret("openweathermap_api_key"), "from-file")
        check("file wins (tavily)",
              config.resolve_tool_secret("tavily_api_key"), "tvly-file")
        check("keychain fills the gap (github)",
              config.resolve_tool_secret("github_token"), "gh-keychain")
        check("absent everywhere is None",
              config.resolve_tool_secret("huggingface_token"), None)

        # -- 2. deleting the file falls back to the keychain --------------
        print("\n=== 2. deleting secrets.yaml falls back to the keychain ===")
        secrets_file.unlink()
        reset({"path": secrets_file})
        check("weather now from keychain",
              config.resolve_tool_secret("openweathermap_api_key"),
              "from-keychain")
        check("no crash when the file is gone", True, True)

        # -- 3. env map omits blanks, never sets "" ----------------------
        print("\n=== 3. blank values are omitted, not exported empty ===")
        secrets_file.write_text(
            'openweathermap_api_key: ""\n'
            'tavily_api_key: "   "\n'
            'github_token: "gh-real"\n'
        )
        config._key_from_keyring = FakeKeyring({})
        reset({"path": secrets_file})
        exported = config.tool_secret_env()
        check("blank weather omitted", "OPENWEATHERMAP_API_KEY" in exported, False)
        check("whitespace tavily omitted", "TAVILY_API_KEY" in exported, False)
        check("real github exported",
              exported.get("GITHUB_TOKEN"), "gh-real")

        # -- 4. YAML retyping -------------------------------------------
        print("\n=== 4. YAML scalar retyping is coerced, not crashed on ===")
        secrets_file.write_text(
            'openweathermap_api_key: 1234567890\n'   # int, not str
            'tavily_api_key: on\n'                  # bool, not str
        )
        reset({"path": secrets_file})
        check("int key survives",
              config.resolve_tool_secret("openweathermap_api_key"), "1234567890")
        check("bool key survives as text",
              config.resolve_tool_secret("tavily_api_key"), "True")

        # -- 5. malformed file is ignored, not fatal --------------------
        print("\n=== 5. malformed YAML is ignored, agent still starts ===")
        secrets_file.write_text("this: is: not: valid: yaml\n\t- broken\n")
        config._key_from_keyring = FakeKeyring({"github_token": "gh-kc"})
        reset({"path": secrets_file})
        check("falls through to keychain",
              config.resolve_tool_secret("github_token"), "gh-kc")
        check("unreadable file does not raise", True, True)

        # -- 6. the real file and the real keychain ---------------------
        print("\n=== 6. live resolution: repo secrets.yaml + real keychain ===")
        config._key_from_keyring = real_keyring   # undo the stand-in first
        reset({"path": Path(config.__file__).resolve().parent / "secrets.yaml"})
        print(f"  path: {config.TOOL_SECRETS_PATH}")
        print(f"  exists: {config.TOOL_SECRETS_PATH.exists()}")
        for line in config.describe_tool_secrets():
            print(f"    {line}")

    finally:
        config._key_from_keyring = real_keyring
        reset({"path": Path(config.__file__).resolve().parent / "secrets.yaml"})

    print("\n--- summary ---")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
