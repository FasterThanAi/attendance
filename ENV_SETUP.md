# Attend — env & keys checklist (Mac setup)

Based on the roadmap's Phase 0 config contract (`services/api/app/config.py`) plus what Phase 8/9 imply but don't spell out.

## What you actually need to get Phase 0 running locally

Nothing here requires a paid API key yet — Postgres, Redis and object storage all run locally via Docker on your Mac. You only start needing real third-party keys once you move to Supabase / a VPS in Phase 9.

| Variable | Where it comes from | Needed for |
|---|---|---|
| `DATABASE_URL` | Local: docker-compose Postgres (user/pass you choose). Later: Supabase dashboard → Project Settings → Database | Every phase |
| `REDIS_URL` | Local: docker-compose Redis, no key needed | Job queue (Phase 2+) |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` | Local: MinIO container (generate creds yourself). Later: Supabase Storage keys or real S3/DigitalOcean Spaces keys | Video/photo storage (Phase 2+) |
| `JOB_DATA_DIR` | Just a path, e.g. `/data/jobs`, on the shared Docker volume | Pipeline artifacts |
| `BIOMETRIC_RETENTION_DAYS` | A number you decide (doc default: 180) | Consent/retention (Phase 0, 9) |
| `JWT_SECRET_KEY` | Generate yourself, not in the doc but required for teacher login | Auth (Phase 8) |
| `INSIGHTFACE_HOME` | A local folder; models auto-download here on first run | ML pipeline (Phase 4+) |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` for local dev | Frontend |

## Mac prerequisites

1. **Docker Desktop for Mac** — runs Postgres, Redis, MinIO, the API and worker containers. Install from docker.com if not already present.
2. **Python 3.11** — `brew install python@3.11` (the ML stack, insightface/onnxruntime, needs this exact minor version per the global prompt).
3. **Node.js 20+** — `brew install node` for the Next.js frontend.
4. **ffmpeg** — `brew install ffmpeg` (used directly via subprocess, per the global prompt's "no moviepy" rule).
5. **Apple Silicon note**: `onnxruntime` on M1/M2/M3 Macs — install `onnxruntime` (CPU) normally; GPU acceleration (CoreML execution provider) is optional and not required to get the pipeline working.

## Generating the secrets locally

```bash
# JWT signing secret
openssl rand -hex 32

# MinIO dev access key / secret key
openssl rand -hex 16   # access key
openssl rand -hex 32   # secret key
```

## When you'll need real third-party keys (not yet)

- **Supabase** (Postgres + Storage) — only once you move off local Docker Postgres/MinIO, typically end of Phase 0 or at Phase 9 deployment. Get `SUPABASE_URL` + service role key from your Supabase project dashboard.
- **A VPS** (Hetzner/DigitalOcean) — Phase 9 only, for running the worker (Vercel can't run it).
- **Vercel** — Phase 9, for the frontend deploy. Free tier, no card needed for a personal project.
- No OpenAI/Anthropic/face-recognition SaaS keys are needed anywhere — the doc deliberately uses open-source InsightFace models that download once and run locally.

## Next step

Fill in `.env` from `.env.example`, then run:
```bash
docker-compose up
```
per Phase 0's definition of done.
