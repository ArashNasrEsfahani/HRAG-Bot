"""Phase 7-B: Math-aware embedder selector tests.

Four tests:
1. dimension_for_model("sentence-transformers/all-mpnet-base-v2") → 768.
2. dimension_for_model("bogus/model") → None.
3. dimension_for_model("BAAI/bge-small-en-v1.5") → 384.
4. `hrag embeddings-list` (via CliRunner) exits 0 and contains "specter2".
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from click.testing import CliRunner  # noqa: E402

from hrag.providers.embeddings import dimension_for_model  # noqa: E402


# ---------------------------------------------------------------------------
# Helper tests for dimension_for_model
# ---------------------------------------------------------------------------


def test_dimension_for_mpnet() -> None:
    """all-mpnet-base-v2 is the default model and must resolve to 768."""
    result = dimension_for_model("sentence-transformers/all-mpnet-base-v2")
    assert result == 768, f"Expected 768, got {result!r}"


def test_dimension_for_bogus_model() -> None:
    """Unknown models not on the curated list must return None."""
    result = dimension_for_model("bogus/model")
    assert result is None, f"Expected None, got {result!r}"


def test_dimension_for_bge_small() -> None:
    """bge-small-en-v1.5 is the 384-dimensional preset."""
    result = dimension_for_model("BAAI/bge-small-en-v1.5")
    assert result == 384, f"Expected 384, got {result!r}"


# ---------------------------------------------------------------------------
# CLI test for `hrag embeddings-list`
# ---------------------------------------------------------------------------


def test_embeddings_list_cli_exits_ok_and_shows_specter2() -> None:
    """`hrag embeddings-list` must exit 0 and include 'specter2' in output."""
    from hrag.cli import cli as cli_group  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(cli_group, ["embeddings-list"])
    assert result.exit_code == 0, (
        f"Expected exit code 0, got {result.exit_code}.\nOutput:\n{result.output}"
    )
    assert "specter2" in result.output, (
        f"'specter2' not found in output:\n{result.output}"
    )
