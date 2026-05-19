"""Visual probe — open taxonomy, click a node, screenshot the popover."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).resolve().parent
SHOT_TREE = OUT_DIR / "probe_tree.png"
SHOT_POPOVER = OUT_DIR / "probe_popover.png"


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
        page.click("#open-taxonomy")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SHOT_TREE))
        print(f"[probe] tree screenshot -> {SHOT_TREE}")

        # Pick a NON-root node card (root has no parent so can't be reparented;
        # but click works on root too). Use the first depth-1 node.
        info = page.evaluate(
            r"""() => {
                const node = document.querySelector('g.tax-node.depth-1');
                if (!node) return { found: false };
                const r = node.getBoundingClientRect();
                const id = node.getAttribute('data-id');
                return {
                    found: true,
                    id,
                    rect: { x: r.x, y: r.y, w: r.width, h: r.height,
                            cx: r.x + r.width/2, cy: r.y + r.height/2 },
                };
            }"""
        )
        print("[probe] target node:", json.dumps(info))
        if not info.get("found"):
            print("[probe] no depth-1 node found")
            page.screenshot(path=str(SHOT_POPOVER))
            browser.close()
            return 2

        page.mouse.click(info["rect"]["cx"], info["rect"]["cy"])
        page.wait_for_timeout(800)
        page.screenshot(path=str(SHOT_POPOVER))
        print(f"[probe] popover screenshot -> {SHOT_POPOVER}")

        # Capture popover state.
        pop = page.evaluate(
            r"""() => {
                const pop = document.getElementById('tax-popover');
                if (!pop) return { found: false };
                const chips = Array.from(pop.querySelectorAll('.tax-doc-chip, [data-role="doc-chip"]'));
                const buttons = Array.from(pop.querySelectorAll('button')).map(b => b.textContent.trim().slice(0,40));
                return {
                    found: true,
                    hidden: pop.hidden,
                    display: getComputedStyle(pop).display,
                    n_chips: chips.length,
                    chip_classes: chips.slice(0,4).map(c => c.className),
                    chip_text_samples: chips.slice(0,4).map(c => (c.textContent||'').trim().slice(0,60)),
                    buttons: buttons.slice(0, 12),
                };
            }"""
        )
        print("[probe] popover state:", json.dumps(pop, indent=2))

        if console_msgs:
            print(f"[probe] console msgs ({len(console_msgs)}):")
            for m in console_msgs[-10:]:
                print(f"  [{m['type']}] {m['text']}")
        if page_errors:
            print(f"[probe] page errors ({len(page_errors)}):")
            for e in page_errors:
                print(f"  {e}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
