"""Unit tests for hrag.retrieval.policy — RetrievalPolicy routing table."""

from __future__ import annotations

import pytest

from hrag.config import IntentConfig
from hrag.intent import Intent
from hrag.retrieval.policy import RetrievalPlan, RetrievalPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> IntentConfig:
    return IntentConfig(personal_top_k=3)


@pytest.fixture
def policy(cfg: IntentConfig) -> RetrievalPolicy:
    return RetrievalPolicy(cfg)


# ---------------------------------------------------------------------------
# Routing table — one row per intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent, expected",
    [
        (
            Intent.GREETING,
            RetrievalPlan(scope="none", top_k_override=None, source_types=None),
        ),
        (
            # PERSONAL searches BOTH documents and episodic memory (so uploaded
            # CVs / notes are found for "find my master GPA"), at its own
            # configured retrieve depth (cfg.personal_top_k, =3 in the fixture).
            Intent.PERSONAL,
            RetrievalPlan(
                scope="full",
                top_k_override=3,
                source_types=["document", "episodic"],
            ),
        ),
        (
            Intent.FACTUAL,
            RetrievalPlan(scope="full", top_k_override=None, source_types=None),
        ),
        (
            Intent.GENERAL,
            RetrievalPlan(scope="none", top_k_override=None, source_types=None),
        ),
        (
            Intent.UNCLEAR,
            RetrievalPlan(scope="ask_clarify", top_k_override=None, source_types=None),
        ),
    ],
)
def test_policy_routing(
    policy: RetrievalPolicy, intent: Intent, expected: RetrievalPlan
) -> None:
    assert policy.plan(intent) == expected


# ---------------------------------------------------------------------------
# top_k_override is read from config, not hard-coded
# ---------------------------------------------------------------------------


def test_personal_top_k_is_configurable() -> None:
    cfg = IntentConfig(personal_top_k=10)
    p = RetrievalPolicy(cfg)
    assert p.plan(Intent.PERSONAL).top_k_override == 10


def test_personal_top_k_default_matches_config_default() -> None:
    """The PERSONAL override tracks IntentConfig.personal_top_k's default."""
    p = RetrievalPolicy(IntentConfig())
    assert p.plan(Intent.PERSONAL).top_k_override == IntentConfig().personal_top_k


# ---------------------------------------------------------------------------
# source_types field correctness
# ---------------------------------------------------------------------------


def test_personal_source_types_spans_documents_and_episodic(
    policy: RetrievalPolicy,
) -> None:
    """PERSONAL searches uploaded documents AND saved memories — not episodic
    alone (that regressed lookups of facts stored in uploaded files)."""
    plan = policy.plan(Intent.PERSONAL)
    assert plan.source_types == ["document", "episodic"]


@pytest.mark.parametrize(
    "intent",
    [Intent.GREETING, Intent.FACTUAL, Intent.GENERAL, Intent.UNCLEAR],
)
def test_non_personal_source_types_is_none(
    policy: RetrievalPolicy, intent: Intent
) -> None:
    """Every non-PERSONAL intent must leave source_types as None (no filter)."""
    plan = policy.plan(intent)
    assert plan.source_types is None, (
        f"intent={intent}: expected source_types=None, got {plan.source_types!r}"
    )


# ---------------------------------------------------------------------------
# Scope values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent, expected_scope",
    [
        (Intent.GREETING, "none"),
        (Intent.PERSONAL, "full"),
        (Intent.FACTUAL,  "full"),
        (Intent.GENERAL,  "none"),
        (Intent.UNCLEAR,  "ask_clarify"),
    ],
)
def test_scope_values(
    policy: RetrievalPolicy, intent: Intent, expected_scope: str
) -> None:
    assert policy.plan(intent).scope == expected_scope


# ---------------------------------------------------------------------------
# RetrievalPlan is frozen (immutable)
# ---------------------------------------------------------------------------


def test_retrieval_plan_is_frozen(policy: RetrievalPolicy) -> None:
    plan = policy.plan(Intent.FACTUAL)
    with pytest.raises((AttributeError, TypeError)):
        plan.scope = "none"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievalPlan equality is value-based
# ---------------------------------------------------------------------------


def test_retrieval_plan_equality() -> None:
    a = RetrievalPlan(scope="full", top_k_override=None, source_types=None)
    b = RetrievalPlan(scope="full", top_k_override=None, source_types=None)
    assert a == b


def test_retrieval_plan_inequality() -> None:
    a = RetrievalPlan(scope="full", top_k_override=None, source_types=None)
    b = RetrievalPlan(scope="episodic", top_k_override=3, source_types=["episodic"])
    assert a != b
