"""Screenshot a Streamlit page after it has actually rendered.

`chromium --screenshot` fires before the websocket has delivered the script's
output, so it captures Streamlit's blank white shell. This drives Chromium over
the DevTools protocol instead: navigate, poll the DOM until the app's own
markup is present, then capture. Used only to eyeball the dashboard; not part of
the app.

Usage:  python tools/shot.py URL OUT.png [--width 1000] [--height 1400]
        [--wait ".fw-row"] [--click "#sendbtn"] ...
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time
import urllib.request

import websockets

PORT = 9333


def launch(width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "chromium", "--headless=new", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", f"--remote-debugging-port={PORT}",
            f"--window-size={width},{height}", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def ws_url() -> str:
    for _ in range(100):
        try:
            pages = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1).read()
            )
            for p in pages:
                if p.get("type") == "page" and p.get("webSocketDebuggerUrl"):
                    return p["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("chromium did not expose a debugging endpoint")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=1400)
    ap.add_argument("--wait", default=".fw-head", help="CSS selector to wait for")
    ap.add_argument("--click", action="append", default=[], help="CSS selector to click first")
    ap.add_argument("--type", action="append", default=[], help="selector=text pairs")
    ap.add_argument("--full", action="store_true", help="capture full page height")
    args = ap.parse_args()

    proc = launch(args.width, args.height)
    try:
        async with websockets.connect(ws_url(), max_size=64 * 1024 * 1024) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            async def js(expr):
                r = await cmd(
                    "Runtime.evaluate",
                    expression=expr,
                    returnByValue=True,
                    awaitPromise=True,
                )
                return r.get("result", {}).get("value")

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await cmd("Page.navigate", url=args.url)

            # Wait for the app's own markup, not merely for load: Streamlit
            # paints a white shell first and fills it over the websocket.
            deadline = time.time() + 60
            while time.time() < deadline:
                found = await js(
                    f"!!document.querySelector({args.wait!r})"
                )
                if found:
                    break
                await asyncio.sleep(0.4)
            else:
                print(f"WARNING: {args.wait} never appeared; capturing anyway")

            for sel in args.click:
                await js(f"document.querySelector({sel!r})?.click()")
                await asyncio.sleep(1.5)

            for pair in args.type:
                sel, _, text = pair.partition("=")
                await js(
                    f"""(() => {{
                      const el = document.querySelector({sel!r});
                      if (!el) return 'no el';
                      const set = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value').set;
                      set.call(el, {text!r});
                      el.dispatchEvent(new Event('input', {{bubbles: true}}));
                      return 'ok';
                    }})()"""
                )
                await asyncio.sleep(0.6)

            await asyncio.sleep(2.0)  # let fonts and the fragment settle

            if args.full:
                h = await js("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
                await cmd("Emulation.setDeviceMetricsOverride",
                          width=args.width, height=int(h or args.height),
                          deviceScaleFactor=1, mobile=False)

            shot = await cmd("Page.captureScreenshot", format="png")
            with open(args.out, "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            print("wrote", args.out)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
