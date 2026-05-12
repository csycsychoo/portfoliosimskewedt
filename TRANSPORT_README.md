# Transport branch — do not merge

This branch is **not part of the existing project**. It only exists as a
transport mechanism to deliver the scaffolded Next.js port to the new repo
`portfoliosimskewedtnextjs`, because the sandbox session that created the files
was not authorized to push there directly.

## What to do with this branch

1. Clone or fetch this branch:
   ```bash
   git fetch origin bootstrap-nextjs-port
   git checkout bootstrap-nextjs-port
   ```
2. Copy `portfoliosimskewedtnextjs/` into the root of your new repo
   `csycsychoo/portfoliosimskewedtnextjs`:
   ```bash
   cp -r portfoliosimskewedtnextjs/. /path/to/csycsychoo-portfoliosimskewedtnextjs/
   cd /path/to/csycsychoo-portfoliosimskewedtnextjs
   git add .
   git commit -m "Bootstrap Next.js port from Streamlit prototype"
   git push origin main  # or your default branch
   ```
3. Delete this branch (it has no purpose after transport):
   ```bash
   git push origin --delete bootstrap-nextjs-port
   git branch -D bootstrap-nextjs-port
   ```

## What's in `portfoliosimskewedtnextjs/`

See `portfoliosimskewedtnextjs/README.md` for the full layout. Summary:
- Refactored Python engine with seed parameter and the B1–B4 fixes from the audit
- 14 passing tests including a golden-master snapshot at seed=42
- Vercel Python serverless handler at `api/simulate.py`
- Next.js App Router UI with Plotly charts and the three presets
- `vercel.json` pinning `@vercel/python@4.3.0`, 60s function timeout
