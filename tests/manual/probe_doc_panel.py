"""Probe: open taxonomy → click leaf with docs → click a doc chip →
verify right-side panel renders with preview."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parent
SHOT = OUT / "probe_doc_panel.png"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_context(viewport={"width": 1400, "height": 900}).new_page()
        msgs = []
        page.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))

        page.goto(URL, wait_until="networkidle")
        page.click("#open-taxonomy")
        page.wait_for_timeout(1200)

        # Find a LEAF node with docs > 0 (skip internal nodes whose docs come
        # from children — clicking those shows zero assigned docs).
        target = page.evaluate(
            r"""() => {
                const nodes = Array.from(document.querySelectorAll('g.tax-node'));
                for (const n of nodes) {
                    const sub = (n.querySelector('text.tax-card-sub')?.textContent || '');
                    if (sub.includes('children')) continue;   // internal node
                    const m = sub.match(/(\d+)\s+docs?/);
                    if (m && parseInt(m[1], 10) > 0) {
                        const r = n.getBoundingClientRect();
                        return { x: r.x + r.width/2, y: r.y + r.height/2,
                                 label: n.querySelector('text.tax-card-label')?.textContent,
                                 sub };
                    }
                }
                return null;
            }"""
        )
        print("target leaf:", target)
        if not target:
            print("no leaf with docs found")
            page.screenshot(path=str(SHOT))
            b.close()
            return 1

        page.mouse.click(target["x"], target["y"])
        page.wait_for_timeout(700)

        # Click the first doc chip title.
        chip_info = page.evaluate(
            r"""() => {
                const title = document.querySelector('#tax-pop-docs .doc-chip-title, #tax-pop-docs button[data-role="title"], #tax-pop-docs [data-role="title"]');
                if (!title) {
                    // Fallback: any chip title-ish text in the popover
                    const t = document.querySelector('#tax-pop-docs .tax-doc-chip-title');
                    if (!t) return null;
                    const r = t.getBoundingClientRect();
                    return { x: r.x + 4, y: r.y + r.height/2, txt: t.textContent };
                }
                const r = title.getBoundingClientRect();
                return { x: r.x + 4, y: r.y + r.height/2, txt: title.textContent };
            }"""
        )
        print("first chip title:", chip_info)
        if not chip_info:
            # Try clicking the "view" magnifier button on the first chip
            page.evaluate(
                r"""() => {
                    const v = document.querySelector('#tax-pop-docs button[data-role="view"], #tax-pop-docs .tax-doc-chip-view');
                    if (v) v.click();
                }"""
            )
        else:
            page.mouse.click(chip_info["x"], chip_info["y"])
        page.wait_for_timeout(1200)

        # Check the doc panel
        st = page.evaluate(
            r"""() => {
                const p = document.getElementById('doc-panel');
                if (!p) return { found: false };
                const r = p.getBoundingClientRect();
                return {
                    found: true,
                    aria_hidden: p.getAttribute('aria-hidden'),
                    transform: getComputedStyle(p).transform,
                    rect: { x: r.x, y: r.y, w: r.width, h: r.height },
                    title: document.getElementById('doc-panel-title')?.textContent,
                    body_text_snippet: (document.getElementById('doc-panel-body')?.textContent || '').slice(0, 120),
                    meta_html: document.getElementById('doc-panel-meta')?.innerHTML.slice(0, 300),
                };
            }"""
        )
        print("doc-panel state:")
        print(json.dumps(st, indent=2))

        page.screenshot(path=str(SHOT))
        print(f"screenshot -> {SHOT}")

        if msgs:
            print("\nconsole tail:")
            for m in msgs[-10:]:
                print(" ", m)

        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
