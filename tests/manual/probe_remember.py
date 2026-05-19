"""Probe the Smart Remember modal (composer button → modal)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).resolve().parent
SHOT = OUT_DIR / "probe_remember.png"


def main() -> int:
    console_msgs: list[dict] = []
    page_errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.on("console", lambda m: console_msgs.append({"type": m.type, "text": m.text}))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(600)

        # Make sure the composer is empty (default state).
        page.evaluate("() => { const el = document.getElementById('input'); if (el) el.value = ''; }")

        # Click the Remember button.
        page.click("#remember-btn")
        page.wait_for_timeout(2500)   # extract endpoint may take a moment

        page.screenshot(path=str(SHOT))
        print(f"[probe] screenshot -> {SHOT}")

        state = page.evaluate(
            r"""() => {
                const modal = document.getElementById('rem-modal');
                const scrim = document.getElementById('rem-modal-scrim');
                const body  = document.getElementById('rem-modal-body');
                const title = document.getElementById('rem-modal-title');
                const sub   = document.getElementById('rem-modal-sub');
                const saveBtn = document.getElementById('rem-modal-save');
                if (!modal) return { found: false };
                const items = Array.from(body?.querySelectorAll('[data-role="rem-item"], .rem-item, label') || []);
                return {
                    found: true,
                    modal_hidden: modal.hidden,
                    modal_display: getComputedStyle(modal).display,
                    scrim_hidden: scrim ? scrim.hidden : null,
                    title: title ? title.textContent.trim() : null,
                    sub: sub ? sub.textContent.trim() : null,
                    n_items: items.length,
                    items_preview: items.slice(0, 4).map(li => (li.textContent || '').trim().slice(0, 80)),
                    save_disabled: saveBtn ? saveBtn.disabled : null,
                };
            }"""
        )
        print("[probe] modal state:")
        print(json.dumps(state, indent=2))

        if console_msgs:
            print(f"[probe] console msgs ({len(console_msgs)}):")
            for m in console_msgs[-15:]:
                print(f"  [{m['type']}] {m['text']}")
        if page_errors:
            print(f"[probe] page errors:")
            for e in page_errors:
                print(f"  {e}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
