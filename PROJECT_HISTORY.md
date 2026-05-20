# FOBO AI — Project History & Session Log

Snapshot exported on **2026-05-20** to `D:\POST\epl` for migration to new development machine.

This file is a faithful reconstruction of every meaningful change to the project,
sourced directly from the git commit log of `https://github.com/jeonhs9110/epl.git`.
No commit was rewritten or squashed — full history is preserved in `.git/`.

---

## Live state at export time

| Item | Value |
|---|---|
| Live frontend | http://34.64.147.124/ |
| Private repo | https://github.com/jeonhs9110/epl |
| Public showcase | https://github.com/jeonhs9110/epl-showcase |
| GCP project | `fobo-ai-jeonhs9110` (asia-northeast3 / Seoul) |
| CPU VM | `fobo-cpu` — `e2-standard-2` (2 vCPU, 8 GB), always-on, runs Flask + scraper |
| GPU VM | spawned on demand (Spot L4 / T4), self-deletes after training |
| Bucket | `gs://fobo-ai-jeonhs9110-shared/` — data + models + history |
| Monthly cost | ~$30/mo (CPU 24/7 + bucket + occasional GPU) |

## Last trained model snapshot

| Artifact | Size | Trained |
|---|---|---|
| `FOBO_LEAGUE_AWARE_current.pth` | 19.8 MB | 2026-04-21 15:12 UTC |
| `xgb_classifier.json` | 1.2 MB | 2026-04-21 15:12 UTC |
| `lgbm_classifier.joblib` | 4.1 MB | 2026-04-21 15:12 UTC |
| `calibrator.joblib` | 975 B  | 2026-04-21 15:12 UTC |
| `ppo_agent.pth` | 14.4 MB | 2026-04-21 15:12 UTC |

Trained on 8,852 matches across 13 European leagues, with the corrected
pipeline order (Hybrid Ensemble → Platt Calibration → PPO).

---

## Commit log — 37 commits, in reverse chronological order

### Phase 5 — RL ordering fix + GPU orchestration bugfixes (2026-04-21)

These came out of an interview where a senior ML reviewer challenged the
pipeline ordering ("RL doesn't come at the end"). Triggered a full
re-evaluation, multiple GPU-orchestration debugging rounds.

- `6129d68` Fix LeagueAwareModel constructor args (takes 2 not 5)
- `66f3a73` gpu_startup.sh: export FOBO_PPO_ONLY from ppo-only metadata
- `eb07274` Fix step 7 'num_teams' KeyError + add PPO-only retraining mode
- `0f81eaf` Fix zombie gunicorn: queue-based CPU stdout streaming + skip_cpu_scrape
- `44124f6` Fix two GPU-spawn bugs exposed by ZONE_RESOURCE_POOL_EXHAUSTED
- `d7e3d6d` CPU VM: set FOBO_SKIP_HISTORICAL_UPDATE=true
- `8291df0` Stream CPU-side scrape stdout into the Update modal terminal
- `4f5ecf0` Run Update (Full) = CPU scrape -> auto-spawn GPU -> train -> self-delete
- `bf50490` Reorder pipeline: Hybrid + Calibration before PPO RL

### Phase 4 — Site polish + Hardening (2026-04-20)

- `f32f5dd` Run Update = scrape-only forever on the CPU VM
- `e01242f` Fix GPU self-shutdown — resolve instance name from metadata server
- `ba9638c` Home: add Results & Honest Limitations section
- `a8d16a1` Trim AI Insights tab to a Project History timeline

### Phase 3 — Chatbot + Bilingual home + Live streaming (2026-04-19, late)

- `c976659` Stream DL + old_matches subprocess stdout to the frontend terminal
- `a581cfc` Chatbot: feed the full FOBO pipeline context into every LLM call
- `5706592` Home tab polish: Result column, GNN insight, softer EN wording
- `468f5ef` cpu_startup.sh: read OPENAI_API_KEY + OPENAI_MODEL from VM metadata
- `cef185e` Frontend polish + OpenAI chatbot + Dixon-Coles BTTS/Over2.5 + bilingual Home
- `de11b6b` Stream raw GPU stdout (workers, per-match) to the frontend terminal

### Phase 2 — Training pipeline portability + bugfixes (2026-04-19, evening)

- `359a07a` Skip re-scraping historical match odds when FOBO_SKIP_HISTORICAL_UPDATE=true
- `aa6afea` Force XGBoost + LightGBM to CPU when FOBO_XGB_CPU=true
- `4e580c3` Always sync scraped CSVs from bucket to CPU VM
- `432bb6d` GPU VM: install Chrome + Xvfb for full-mode scraping
- `b45510e` Fix GPU startup pip install on Ubuntu 22.04
- `56772df` Stream GPU training progress to CPU VM for live terminal display
- `d732177` Skip PPO fallback training on CPU startup when FOBO_SKIP_TRAINING=true
- `a45c517` Fix CPU Docker build: install CPU-only torch before requirements

### Phase 1 — Cloud deployment scaffolding (2026-04-19)

- `d1f1677` Add model backup on CPU VM: 15-min GCS pull + admin reload endpoint
- `f9632c6` Add daily scrape scheduler (11:00 KST) + scrape-only pipeline mode
- `cb83e7f` Add GCP deployment scaffolding for two-VM architecture
- `dd95a47` Add Update Mode: one-shot refresh pipeline with live progress UI

### Phase 0 — Pre-cloud baseline (2026-04-10 through 2026-04-12)

- `caa54d7` Add README with quick start instructions for home setup
- `15819ec` FOBO AI - Full pipeline with trained models
- `6133427` Update match results, retrain model, and refresh predictions
- `cd6d420` Add trained model weights and encoders for portable deployment
- `00d3ab0` Fix /train_rl route: use PPOAgent/train_ppo_agent, save to correct path
- `1e57292` Refactor: split monolithic index.html into Jinja2 partials; fix ML pipeline ordering bugs
- `930dcf7` Initial commit: FOBO AI football prediction pipeline

---

## Architectural decisions worth re-reading

### Correct pipeline order (after Phase 5 review)

```
① Scraping (Flashscore, 13 leagues, parallel Chrome workers via Xvfb)
② Validation (check_data.py — CSV integrity)
③ Sequence Optimisation (per-league LSTM look-back window)
④ DL Training (LeagueAwareModel = Transformer + GAT + Dixon-Coles NLL)
⑤ Hybrid Ensemble (XGBoost + LightGBM on DL embeddings)
⑥ Calibration (Platt scaling)
⑦ PPO RL Decision Agent (Kelly-shaped reward on CALIBRATED probabilities)
⑧ Flask API + frontend
```

The earlier order had PPO between DL and Hybrid, which would have trained PPO
on miscalibrated probabilities → biased Kelly-Criterion bet sizing → polluted
policy. Stage ⑦ now runs AFTER ⑥ Calibration so the agent learns on top of
finalized probabilities.

### Two-VM cloud architecture

- **CPU VM (always-on)** — `e2-standard-2` (Seoul), serves Flask 24/7 + Selenium
  scraping in Docker (Chrome + chromedriver + Xvfb baked in). Scheduled daily
  11 KST scrape. ~$24/mo.
- **GPU VM (on-demand Spot)** — created by CPU VM via Compute Engine REST API,
  pulls fresh data from bucket, trains, pushes models back, self-deletes.
  ~$0.40 per training run.
- **Shared Cloud Storage bucket** — single source of truth. CPU pulls every
  15 min + on /admin/reload_models.

### Key env vars that gate behavior

| Var | Purpose |
|---|---|
| `FOBO_CLOUD` | Activate Linux-Docker-Chrome browser flags |
| `FOBO_GCS_BUCKET` | Enables storage_sync; unset → all local |
| `FOBO_TEST_MODE` | Smoke-test path (1 epoch, skip scrape) |
| `FOBO_SKIP_TRAINING` | App boots without retraining |
| `FOBO_SKIP_HISTORICAL_UPDATE` | Don't re-scrape pre-existing CSVs |
| `FOBO_SKIP_DL_PPO` | Skip the internal PPO inside train_dl.py |
| `FOBO_SKIP_SCRAPE` | GPU VM skips steps 1-3 (data is fresh in bucket) |
| `FOBO_PPO_ONLY` | GPU VM only retrains PPO (~25 min) |
| `FOBO_XGB_CPU` | Force XGBoost/LightGBM to CPU (CUDA segfaults on some images) |
| `FOBO_SCHEDULE_DAILY` | APScheduler runs daily 11 KST scrape |
| `FOBO_ADMIN_TOKEN` | Auth for /admin/reload_models, /admin/training_log, /admin/training_progress |
| `OPENAI_API_KEY` | Activate chatbot (per-match report + general questions) |

---

## Resuming on the new machine

```bash
cd D:\POST\epl
git status                 # should show working tree clean
git log --oneline -5       # should match commits above

# Optional: pull anything pushed from the cloud VMs since this snapshot
git pull

# Local Python deps (if you want to run the Flask app locally)
pip install -r requirements.txt

# Run locally (uses CSVs in this folder, no GCP needed)
python run_pipeline.py     # interactive CLI
# OR
python app.py              # Flask on :5000

# Cloud operations (one-time auth on new machine)
gcloud auth login
gcloud config set project fobo-ai-jeonhs9110
gcloud compute instances list    # confirm fobo-cpu is RUNNING
```

The OpenAI key and GCP admin token live in VM metadata (not in this repo).
Copy from the current machine's environment if needed; otherwise the cloud VMs
already have them.
