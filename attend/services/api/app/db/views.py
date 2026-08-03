"""Phase 8 deliverable 4: the `current_attendance` view.

`AttendanceRecord` is append-only (non-negotiable rule 4) -- a correction is
a NEW row with `supersedes_id` pointing at the row it corrects, never an
UPDATE. This view resolves any such correction chain, of any depth, down to
exactly the one row per (class_session_id, student_id) that has no
successor -- "the row nothing else points at via supersedes_id" -- which by
construction is the most recent decision for that student in that session.
Every query that needs "what is this student's attendance right now" must
go through this view, never a raw `attendance_record` query ordered by
`created_at DESC LIMIT 1` (that would silently break if a correction were
ever inserted with a backdated or out-of-order `created_at`, which append-
only history makes entirely possible over time).

Defined ONCE, here, and wired into two places that must never drift apart:
  - the Alembic migration that creates it against real Postgres
    (services/api/alembic/versions/0004_current_attendance_view.py), for
    the real deployed database.
  - an SQLAlchemy `after_create`/`before_drop` event on `Base.metadata`, so
    `Base.metadata.create_all()` -- what services/api/tests/conftest.py's
    in-memory SQLite fixture uses instead of running Alembic (see that
    fixture's own docstring/comments) -- creates the exact same view. Without
    this, a test asserting on `current_attendance` would either error (view
    doesn't exist) or, worse, silently test against handwritten
    supersede-chain-walking Python instead of the actual view everything
    else in production relies on.

Plain SQL (LEFT JOIN + IS NULL), not a recursive CTE or window function,
because it has to run unmodified on both SQLite (tests) and Postgres
(real) -- and doesn't need recursion anyway: "no other row points at me via
supersedes_id" is a single join test, regardless of how many corrections
came before the current tip of the chain.

Importing this module registers the event listeners as a side effect; it is
imported once, here, by `app/db/__init__.py`, so anything that imports
`app.db.models` (which is everywhere in this codebase) transitively gets the
view registered before `Base.metadata.create_all()` ever runs.
"""

from __future__ import annotations

from sqlalchemy import DDL, event

from app.db.models import Base

CURRENT_ATTENDANCE_VIEW_NAME = "current_attendance"

_CREATE_VIEW_SQL = f"""
CREATE VIEW {CURRENT_ATTENDANCE_VIEW_NAME} AS
SELECT ar.*
FROM attendance_record ar
LEFT JOIN attendance_record newer ON newer.supersedes_id = ar.id
WHERE newer.id IS NULL
"""

_DROP_VIEW_SQL = f"DROP VIEW IF EXISTS {CURRENT_ATTENDANCE_VIEW_NAME}"

create_current_attendance_view_ddl = DDL(_CREATE_VIEW_SQL)
drop_current_attendance_view_ddl = DDL(_DROP_VIEW_SQL)

# Registered against Base.metadata (not a specific Table) because a VIEW,
# unlike a table, has no ORM-mapped class of its own to attach the DDL to --
# "after every table in this metadata has been created" is exactly the
# right hook, since the view selects from attendance_record and must run
# after that table exists.
event.listen(Base.metadata, "after_create", create_current_attendance_view_ddl)
event.listen(Base.metadata, "before_drop", drop_current_attendance_view_ddl)
