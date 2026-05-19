"""Headless-browser probe for the taxonomy editor — deep CSS diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = "http://127.0.0.1:8000"
OUT_DIR = Path(__file__).resolve().parent
SCREENSHOT = OUT_DIR / "taxonomy_probe.png"


def main() -> int:
    console_msgs: list[dict] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.on("console", lambda m: console_msgs.append({"type": m.type, "text": m.text}))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        print(f"[probe] loading {URL} ...", flush=True)
        page.goto(URL, wait_until="networkidle")

        page.wait_for_selector("#open-taxonomy", timeout=8000)
        page.click("#open-taxonomy")
        page.wait_for_timeout(1200)

        info = page.evaluate(
            r"""() => {
                const svg = document.getElementById('tax-svg');
                const r = svg ? svg.getBoundingClientRect() : null;
                const cs = svg ? getComputedStyle(svg) : null;

                // Walk up the parent chain and report computed display.
                const ancestors = [];
                let n = svg;
                while (n && n !== document.documentElement) {
                    const c = getComputedStyle(n);
                    const br = n.getBoundingClientRect();
                    ancestors.push({
                        tag: n.tagName.toLowerCase(),
                        id: n.id || null,
                        cls: n.className?.baseVal ?? n.className,
                        display: c.display,
                        visibility: c.visibility,
                        opacity: c.opacity,
                        width: c.width,
                        height: c.height,
                        rect: { w: br.width, h: br.height },
                        hidden_attr: n.hasAttribute('hidden'),
                        inline_display: n.style?.display ?? null,
                    });
                    n = n.parentElement;
                }

                return {
                    svg_hidden_attr: svg ? svg.hasAttribute('hidden') : null,
                    svg_hidden_prop: svg ? svg.hidden : null,
                    svg_computed_display: cs?.display,
                    svg_inline_display: svg?.style?.display ?? null,
                    svg_rect: r ? { w: r.width, h: r.height } : null,
                    svg_outerHTML_head: svg ? svg.outerHTML.slice(0, 200) : null,
                    n_nodes: svg ? svg.querySelectorAll('g.tax-node').length : 0,
                    ancestors,
                };
            }"""
        )

        page.screenshot(path=str(SCREENSHOT))

        print("\n--- SVG state ---")
        print(json.dumps(info, indent=2)[:5000])

        if console_msgs:
            print(f"\n--- console ({len(console_msgs)}) ---")
            for m in console_msgs[-20:]:
                print(f"  [{m['type']}] {m['text']}")
        if page_errors:
            print(f"\n--- page errors ({len(page_errors)}) ---")
            for e in page_errors:
                print(f"  {e}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
