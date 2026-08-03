# Importing app.db.views registers the current_attendance view's DDL
# against Base.metadata's after_create/before_drop events (see that
# module's docstring) -- done here, in the package __init__, so anything
# that imports any app.db submodule (app.db.models, app.db.session, ...)
# transitively gets the view registered before Base.metadata.create_all()
# ever runs, including in services/api/tests/conftest.py's in-memory
# SQLite fixture.
from app.db import views  # noqa: F401
