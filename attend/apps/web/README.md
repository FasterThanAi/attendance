# apps/web

Next.js 14 (App Router, TypeScript, Tailwind). Phase 2 built the upload flow
(`/record`); the rest of the screens (sign in, today, review, reports) come
in Phases 8-9.

## Setup

```bash
cd apps/web
npm install
cp .env.local.example .env.local
npm run dev
```

Then open http://localhost:3000/record -- it needs the api running
(`docker-compose up` from the repo root) to actually upload anything.

## Not verified from the sandbox this was built in

`npm install` could not be run here (no registry access), so this has been
reviewed carefully but never actually compiled or run through `next dev`.
Run `npm install && npm run dev` and `npm run build` on your Mac and tell me
if either surfaces a type error or dependency issue -- the most likely spot
is a version mismatch in package.json, which is a quick fix.
