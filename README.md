# Face Match Verification API

A local, no-API-key face-match service for KYC-style checks (selfie vs. ID document photo).
Runs entirely on your machine using InsightFace (ONNX Runtime) — no per-call cost, no cloud calls.

No frontend needed — this is a pure API your senior (or any other service) can call directly.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Configure environment

Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

`.env` (not committed to git — see [Repo hygiene](#repo-hygiene) below):

```
HOST=0.0.0.0
PORT=8000
```

## 3. Start the API server

Either of these works — both read `HOST`/`PORT` from `.env`:

```bash
# via uvicorn CLI
python -m uvicorn server_insightface:app --host 0.0.0.0 --port 8000

# or directly (uses HOST/PORT from .env)
python server_insightface.py
```

The model loads once at startup (a few seconds), then every `/verify` call is fast (~1-2s).

Interactive API docs (Swagger UI) are auto-generated at:
`http://127.0.0.1:8000/docs`

## 4. Call the API

### Endpoint
`POST /verify` — multipart form upload

| Field / Param        | Type          | Required | Default | Notes                                  |
|-----------------------|---------------|----------|---------|-----------------------------------------|
| `selfie`              | file          | yes      | -       | jpg / png                               |
| `id_doc`               | file          | yes      | -       | jpg / png / pdf                         |
| `approve_threshold`   | query float   | no       | 0.55    | cosine similarity ≥ this → `approve`    |
| `review_threshold`    | query float   | no       | 0.35    | between this and approve → `manual_review` |
| `benchmark`           | query bool    | no       | false   | prints timing breakdown server-side     |

### curl example

```bash
curl -X POST "http://127.0.0.1:8000/verify" \
  -F "selfie=@selfie.jpg" \
  -F "id_doc=@id_card.pdf"
```

### Python example

See `client_example.py`:

```bash
python client_example.py --selfie selfie.jpg --id-doc id_card.pdf
```

### Example response

```json
{
  "decision": "approve",
  "verified": true,
  "confidence_pct": 78.42,
  "cosine_similarity": 0.7842,
  "approve_threshold": 0.55,
  "review_threshold": 0.35,
  "model": "buffalo_l (InsightFace / ONNX Runtime)"
}
```

`decision` is one of `"approve"`, `"manual_review"`, `"reject"`.

## Repo hygiene

`.gitignore` is already set up to keep the following **out** of the repo:

| Excluded                          | Why                                                              |
|------------------------------------|-------------------------------------------------------------------|
| `.env`                             | Local config (host/port). `.env.example` is committed as a template instead. |
| `__pycache__/`, `*.pyc`, `venv/`   | Generated / environment-specific, not code.                      |
| `.insightface/`, `**/models/`      | Model weights are auto-downloaded on first run — large binaries, don't belong in git. |
| `*.jpg`, `*.jpeg`, `*.png`, `*.pdf`, `test_images/`, `sample_data/` | **Biometric/PII test data** (selfies, ID docs) used while testing — never commit real face or ID images to a repo, public or private. |

If you need to keep a non-sensitive image (e.g. a diagram in the README), either add it under an explicit exception in `.gitignore` (e.g. `!docs/*.png`) or rename it into a folder you control.

## Pushing to GitHub

From inside the project folder:

```bash
git init
git add .
git status          # sanity check: confirm .env and any test images are NOT listed
git commit -m "Initial commit: face match verification API"

# create the repo on GitHub first (via github.com or `gh repo create`), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If you'd rather create the GitHub repo from the terminal (requires [GitHub CLI](https://cli.github.com/)):

```bash
gh repo create <repo-name> --private --source=. --remote=origin --push
```

Double-check `git status` before your first commit — if `.env` or any test images show up as tracked, `.gitignore` won't retroactively untrack files that were already committed. In that case run `git rm --cached .env` (and similarly for any image files) before committing.

## Notes for whoever consumes this API

- Thresholds are starting points, not validated — calibrate `approve_threshold` /
  `review_threshold` against a real labeled selfie-vs-ID dataset before trusting decisions
  in production.
- The largest detected face in each image is used (guards against false matches from
  logos/holograms on ID documents).
- This is a POC: no auth, no rate limiting, no persistence of uploaded images (files are
  written to a temp dir per request and deleted immediately after). Add auth + input
  size limits before exposing this outside localhost.
- Biometric data handling/retention needs a compliance review before any real deployment.