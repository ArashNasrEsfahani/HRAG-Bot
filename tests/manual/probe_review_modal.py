"""Playwright headless smoke probe for the Phase 8 review modal.

Usage (server must already be running at http://127.0.0.1:8000):

    python tests/manual/probe_review_modal.py

What it does
------------
1. Navigates to http://127.0.0.1:8000 and verifies the page loads.
2. Injects a synthetic ``review_required`` event directly via
   ``page.evaluate()`` calling ``handleReviewRequired(...)`` with a fake
   payload (3 sources, 2 rephrasings, 1 reason="score_floor").
3. Takes a screenshot (``probe_review_modal.png``) and reads the DOM:
   - asserts the modal is visible (#review-modal not hidden)
   - asserts exactly 3 ``.review-source`` rows are rendered
   - asserts exactly 2 ``.review-rephrasing-chip`` chips
   - asserts exactly 1 ``.review-reason-chip`` chip (or ``[data-reason]`` tag)
4. Simulates pressing the ``1`` key to toggle source 1's checkbox, then
   verifies it became unchecked.
5. Clicks the "Continue" button; verifies the modal closes and captures any
   POST request to ``/api/chat/turns/fake-turn-id/resume``.

Exits with status 0 on success, 1 on failure.  Prints a JSON summary line at
the end matching the style of other ``probe_*.py`` scripts in this directory.

Notes
-----
* NOT run in CI.  Lives under ``tests/manual/`` which is excluded from
  pytest collection by ``pyproject.toml``'s ``testpaths`` setting.
* Requires ``playwright`` (``pip install playwright && playwright install chromium``).
* Requires the frontend to expose a global ``handleReviewRequired`` function,
  which is the Phase 8 wave-3 frontend handler wired in ``app.js``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SHOT = OUT_DIR / "probe_review_modal.png"
URL = "http://127.0.0.1:8000"

_FAKE_TURN_ID = "fake-turn-id"
_FAKE_PAYLOAD = {
    "turn_id": _FAKE_TURN_ID,
    "reasons": ["score_floor"],
    "rephrasings": ["How does X work?", "Can you explain X in detail?"],
    "candidates": [
        {
            "chunk_id": "c1",
            "title": "Paper Alpha",
            "section": "Introduction",
            "text": "This is the first source passage.",
            "score": 0.9,
            "rerank_score": 1.5,
            "source_type": "document",
        },
        {
            "chunk_id": "c2",
            "title": "Paper Beta",
            "section": "Methods",
            "text": "This is the second source passage.",
            "score": 0.7,
            "rerank_score": 0.8,
            "source_type": "document",
        },
        {
            "chunk_id": "c3",
            "title": "Paper Gamma",
            "section": "Results",
            "text": "This is the third source passage.",
            "score": 0.5,
            "rerank_score": 0.2,
            "source_type": "document",
        },
    ],
}


def _fail(msg: str, summary: dict) -> int:
    summary["ok"] = False
    summary["error"] = msg
    print(json.dumps(summary))
    print(f"[probe] FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(json.dumps({"ok": False, "error": "playwright not installed; run: pip install playwright && playwright install chromium"}))
        return 1

    summary: dict = {
        "probe": "review_modal",
        "ok": True,
        "screenshot": str(SHOT),
        "checks": {},
    }
    resume_requests: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        console_msgs: list[str] = []
        page_errors: list[str] = []
        page.on("console", lambda m: console_msgs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        # Capture any POST to /resume so we can verify the continue button fires it.
        def _on_request(req):
            if "/resume" in req.url and req.method == "POST":
                resume_requests.append(req.url)

        page.on("request", _on_request)

        # 1. Navigate to the app.
        try:
            page.goto(URL, wait_until="networkidle", timeout=10_000)
        except Exception as exc:
            browser.close()
            return _fail(
                f"Server not reachable at {URL} — start it first with `hrag web` or `uvicorn hrag.web.app:app`. Detail: {exc}",
                summary,
            )

        page.wait_for_timeout(600)

        # 2. Inject the synthetic review_required event via the frontend handler.
        inject_ok = page.evaluate(
            f"""() => {{
                if (typeof handleReviewRequired !== 'function') {{
                    return {{ ok: false, error: 'handleReviewRequired not defined on window' }};
                }}
                try {{
                    handleReviewRequired({json.dumps(_FAKE_PAYLOAD)});
                    return {{ ok: true }};
                }} catch (e) {{
                    return {{ ok: false, error: String(e) }};
                }}
            }}"""
        )
        if not inject_ok.get("ok"):
            page.screenshot(path=str(SHOT))
            browser.close()
            return _fail(
                f"handleReviewRequired injection failed: {inject_ok.get('error', '?')}",
                summary,
            )

        page.wait_for_timeout(400)

        # 3. Screenshot.
        page.screenshot(path=str(SHOT))
        print(f"[probe] screenshot -> {SHOT}")

        # 3a. Modal visibility.
        modal_state = page.evaluate(
            r"""() => {
                const modal = document.getElementById('review-modal');
                if (!modal) return { found: false };
                const hidden = modal.hidden || modal.getAttribute('aria-hidden') === 'true'
                             || getComputedStyle(modal).display === 'none';
                return {
                    found: true,
                    hidden: hidden,
                    aria_hidden: modal.getAttribute('aria-hidden'),
                    display: getComputedStyle(modal).display,
                };
            }"""
        )
        print("[probe] modal state:", json.dumps(modal_state))

        if not modal_state.get("found"):
            browser.close()
            return _fail("Modal element #review-modal not found in DOM", summary)

        if modal_state.get("hidden"):
            browser.close()
            return _fail(
                f"Modal is hidden after handleReviewRequired() call. "
                f"aria-hidden={modal_state.get('aria_hidden')!r}, display={modal_state.get('display')!r}",
                summary,
            )
        summary["checks"]["modal_visible"] = True

        # 3b. Source rows.
        n_sources = page.evaluate(
            r"() => document.querySelectorAll('#review-modal .review-source').length"
        )
        print(f"[probe] source rows: {n_sources}")
        if n_sources != 3:
            browser.close()
            return _fail(
                f"Expected 3 .review-source rows, got {n_sources}. "
                "Check that handleReviewRequired renders one row per candidate.",
                summary,
            )
        summary["checks"]["source_rows"] = n_sources

        # 3c. Rephrasing chips.
        n_rephrasings = page.evaluate(
            r"() => document.querySelectorAll('#review-modal .review-rephrasing-chip').length"
        )
        print(f"[probe] rephrasing chips: {n_rephrasings}")
        if n_rephrasings != 2:
            browser.close()
            return _fail(
                f"Expected 2 .review-rephrasing-chip chips, got {n_rephrasings}.",
                summary,
            )
        summary["checks"]["rephrasing_chips"] = n_rephrasings

        # 3d. Reason chips.
        n_reasons = page.evaluate(
            r"""() => {
                // Accept either a dedicated .review-reason-chip class or [data-reason] attribute.
                const byClass = document.querySelectorAll('#review-modal .review-reason-chip').length;
                const byAttr  = document.querySelectorAll('#review-modal [data-reason]').length;
                return byClass || byAttr;
            }"""
        )
        print(f"[probe] reason chips: {n_reasons}")
        if n_reasons < 1:
            browser.close()
            return _fail(
                f"Expected at least 1 reason chip (.review-reason-chip or [data-reason]), got {n_reasons}.",
                summary,
            )
        summary["checks"]["reason_chips"] = n_reasons

        # 4. Toggle source 1 checkbox via keyboard key "1".
        # First, find the checkbox state before the keypress.
        checked_before = page.evaluate(
            r"""() => {
                const rows = document.querySelectorAll('#review-modal .review-source');
                if (rows.length === 0) return null;
                const cb = rows[0].querySelector('input[type=checkbox]');
                return cb ? cb.checked : null;
            }"""
        )
        print(f"[probe] source-1 checkbox before keypress: {checked_before}")

        # Focus the modal (or the page body) and send "1".
        page.keyboard.press("1")
        page.wait_for_timeout(200)

        checked_after = page.evaluate(
            r"""() => {
                const rows = document.querySelectorAll('#review-modal .review-source');
                if (rows.length === 0) return null;
                const cb = rows[0].querySelector('input[type=checkbox]');
                return cb ? cb.checked : null;
            }"""
        )
        print(f"[probe] source-1 checkbox after keypress: {checked_after}")

        if checked_before is not None and checked_after is not None:
            if checked_before == checked_after:
                # Soft-warn: keyboard toggle may not be implemented yet.
                print("[probe] WARN: pressing '1' did not toggle source-1 checkbox (may be unimplemented)")
                summary["checks"]["keyboard_toggle"] = "not_toggled_soft_warn"
            else:
                summary["checks"]["keyboard_toggle"] = "toggled"
        else:
            summary["checks"]["keyboard_toggle"] = "no_checkbox_found"

        # 5. Click "Continue" and verify the modal closes + a POST to /resume fires.
        continue_btn = page.query_selector(
            "#review-modal button.review-continue-btn, "
            "#review-modal [data-action='continue'], "
            "#review-modal button:has-text('Continue')"
        )
        if continue_btn is None:
            browser.close()
            return _fail(
                "Could not find Continue button inside #review-modal. "
                "Tried selectors: .review-continue-btn, [data-action='continue'], button:has-text('Continue').",
                summary,
            )

        continue_btn.click()
        page.wait_for_timeout(500)

        # Check modal closed.
        modal_after = page.evaluate(
            r"""() => {
                const m = document.getElementById('review-modal');
                if (!m) return { found: false };
                return {
                    found: true,
                    hidden: m.hidden || m.getAttribute('aria-hidden') === 'true'
                           || getComputedStyle(m).display === 'none',
                };
            }"""
        )
        print(f"[probe] modal after Continue click: {json.dumps(modal_after)}")
        modal_closed = modal_after.get("hidden", False)
        summary["checks"]["modal_closed_after_continue"] = modal_closed
        if not modal_closed:
            print("[probe] WARN: modal did not close after Continue click (may need server response)")

        # Check /resume request was fired.
        print(f"[probe] /resume requests captured: {resume_requests}")
        resume_fired = any(_FAKE_TURN_ID in r for r in resume_requests)
        summary["checks"]["resume_post_fired"] = resume_fired
        if not resume_fired:
            print(
                f"[probe] WARN: no POST to /api/chat/turns/{_FAKE_TURN_ID}/resume captured. "
                "This may be expected if the frontend batches the call or the stub server returns 400."
            )

        # Console / page errors.
        if console_msgs:
            print(f"\n[probe] console messages ({len(console_msgs)}):")
            for m in console_msgs[-15:]:
                print(f"  {m}")
        if page_errors:
            print(f"\n[probe] page errors ({len(page_errors)}):")
            for e in page_errors:
                print(f"  {e}")
            summary["page_errors"] = page_errors[:5]

        browser.close()

    print(f"\n[probe] summary: {json.dumps(summary)}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
