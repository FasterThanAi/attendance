# Attend

A classroom attendance system: a teacher records a ~60s 4K video panning
across the room, the system detects and clusters faces, matches them against
an enrolled student gallery, and produces a draft attendance list for the
teacher to confirm. See `Attend-Classroom-Video-Attendance-Roadmap.pdf` (in
the parent folder) for the full 11-phase engineering plan this repo follows.

This repo currently implements **Phase 0 — Foundations and consent** only.

## What's here

```
attend/
  apps/web/              Next.js frontend (placeholder, starts Phase 2)
  services/api/          FastAPI backend
    app/
      main.py            entrypoint, request-id middleware, /health
      config.py           all env-driven settings (one place, per the brief)
      db/models.py        full 16-table schema
      db/session.py        async engine/session
      routers/health.py
      schemas/            typed Pydantic request/response models
      services/consent.py  assert_consent_valid() -- the consent gate
    alembic/              async-configured migrations
    tests/               pytest + in-memory SQLite fixture
  services/worker/        RQ worker (pipeline stages stubbed, filled in
                          Phases 3-6)
  docker-compose.yml      postgres16 + redis7 + api + worker
  docs/CONSENT.md         plain-English + Tamil-placeholder consent form
```

## How to verify (Mac)

Prerequisites: Docker Desktop, and that's actually it for this phase --
everything else runs inside containers.

```bash
cd attend
cp .env.example .env        # fill in S3_ACCESS_KEY/S3_SECRET_KEY if you've
                             # already got MinIO or Supabase Storage set up;
                             # empty is fine for now, storage isn't used until
                             # Phase 2
docker-compose up --build
```

Expected: four containers come up (postgres, redis, api, worker), with
postgres and redis reporting healthy before api/worker start.

In a second terminal, apply the migration and hit the health endpoint:

```bash
docker-compose exec api alembic upgrade head
curl http://localhost:8000/health
```

Expected `/health` response: `{"service":"attend-api","version":"0.1.0","environment":"development","database":{"ok":true,...},"redis":{"ok":true,...}}`

Run the test suite (uses an in-memory SQLite DB, not the Postgres container,
so it's fast and needs no setup beyond the installed requirements):

```bash
docker-compose exec api pytest -v
```

Expected: all tests in `tests/test_models.py` (one per table) and
`tests/test_consent.py` (consent gate) pass.

## Definition of done (from the roadmap, Phase 0)

- [x] `docker-compose up` starts cleanly, with healthchecks
- [x] `alembic upgrade head` applies (initial migration covers all 16 tables)
- [x] `/health` returns 200 with db + redis connectivity
- [ ] You have a signed consent form template — `docs/CONSENT.md` is the
      template; it still needs your institution's name/contact filled in,
      the Tamil section translated by a fluent speaker, and your supervisor
      to actually see it. That last part is on you, not something I can do.

## What I couldn't verify from here

This was built and syntax-checked (`python -m py_compile` on every file,
`docker-compose.yml` parsed and structurally checked) in a sandbox with no
Docker and no PyPI access, so it has **not** been run end-to-end against a
real Postgres container yet. Run the "How to verify" steps above on your Mac
before treating Phase 0 as done — if `docker-compose up` or `alembic upgrade
head` surface an issue, tell me the error and I'll fix it directly.

## Next: Phase 1

Enrollment and gallery construction (`params.py`'s real content, `embed.py`,
the pose-bucketing enrollment job, `gallery_sanity.py`). Say "start phase 1"
when Phase 0 is verified on your machine.
