"""GUI module smoke tests.

These do NOT spin up the Streamlit server — that would require a real
browser-side runtime. They only verify:
  * Every GUI Python file parses and imports.
  * The CLI exposes the `gui` subcommand.
  * The state helpers expose the expected names.

Streamlit is optional (`[gui]` extra). The whole module is skipped when
the dep isn't installed so the suite still passes on minimal installs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit", reason="streamlit not installed; install with `pip install -e .[gui]`")
pytest.importorskip("pandas", reason="pandas not installed")


def test_gui_module_imports():
    from hrag.gui import state  # noqa: F401

    assert hasattr(state, "get_orchestrator")
    assert hasattr(state, "current_user_id")
    assert hasattr(state, "set_user_id")
    assert hasattr(state, "sidebar_user_pill")


def test_gui_app_file_parses():
    import pathlib
    import ast

    app_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "hrag" / "gui" / "app.py"
    ast.parse(app_path.read_text(encoding="utf-8"))


def test_gui_pages_all_parse():
    import pathlib
    import ast

    pages_dir = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src" / "hrag" / "gui" / "pages"
    )
    page_files = sorted(pages_dir.glob("*.py"))
    assert len(page_files) >= 7, f"expected ≥7 pages, found {len(page_files)}"
    for f in page_files:
        ast.parse(f.read_text(encoding="utf-8"))


def test_cli_exposes_gui_command():
    from click.testing import CliRunner

    from hrag.cli import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert "gui" in result.output, "Expected `gui` command in CLI help output"
    result = CliRunner().invoke(cli, ["gui", "--help"])
    assert "Streamlit" in result.output
