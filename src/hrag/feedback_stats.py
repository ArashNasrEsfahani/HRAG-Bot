"""Shared feedback-statistics helper.

Extracted so both the CLI (``hrag feedback-stats``) and the web API
(``GET /api/feedback/stats``) can call the same pure SQL function without
importing Click or Rich.
"""

from __future__ import annotations


def feedback_summary(db) -> dict:
    """Return aggregate feedback statistics.

    Pure function — no I/O beyond the DB queries; callers handle presentation.

    Parameters
    ----------
    db:
        An ``hrag.db.connection.Database`` instance (or any object that
        exposes ``.execute(sql, params=())`` returning rows accessible by
        column name).

    Returns
    -------
    dict with keys:
        thumbs_up   : int  — rows with rating == +1
        thumbs_down : int  — rows with rating == -1
        total       : int  — thumbs_up + thumbs_down (rating 0 excluded)
        sessions    : int  — distinct session_ids that have any feedback
        top_negative: list[dict]  — up to 5 negatively-rated items, each
                          with {question, session_id, created_at}
    """
    up_row = db.execute(
        "SELECT COUNT(*) AS n FROM feedback WHERE rating = 1"
    ).fetchone()
    down_row = db.execute(
        "SELECT COUNT(*) AS n FROM feedback WHERE rating = -1"
    ).fetchone()
    sessions_row = db.execute(
        "SELECT COUNT(DISTINCT session_id) AS n FROM feedback WHERE rating != 0"
    ).fetchone()

    thumbs_up = up_row["n"] if up_row else 0
    thumbs_down = down_row["n"] if down_row else 0
    sessions = sessions_row["n"] if sessions_row else 0

    # For each negative feedback row find the immediately-preceding user
    # message in the same session.
    neg_rows = db.execute(
        """
        SELECT f.message_id, f.session_id, f.created_at
        FROM feedback f
        WHERE f.rating = -1
        ORDER BY f.created_at DESC
        LIMIT 5
        """
    ).fetchall()

    top_negative: list[dict] = []
    for row in neg_rows:
        user_msg_row = db.execute(
            """
            SELECT content FROM messages
            WHERE session_id = ? AND role = 'user'
              AND message_id < CAST(? AS INTEGER)
            ORDER BY message_id DESC
            LIMIT 1
            """,
            (row["session_id"], row["message_id"]),
        ).fetchone()
        question = user_msg_row["content"] if user_msg_row else "(question not found)"
        top_negative.append(
            {
                "question": question,
                "session_id": row["session_id"],
                "created_at": row["created_at"] or "",
            }
        )

    return {
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "total": thumbs_up + thumbs_down,
        "sessions": sessions,
        "top_negative": top_negative,
    }
