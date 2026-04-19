"""
Unified Update Pipeline for FOBO AI.

Runs the full refresh sequence end-to-end, non-interactively:
  1. Update historical-match odds
  2. Scrape current-season results for all leagues
  3. Scrape upcoming fixtures (next 30 days)
  4. Validate data integrity
  5. Retrain deep learning model
  6. Retrain hybrid XGBoost ensemble
  7. Retrain PPO reinforcement-learning agent

Callers:
  - run_pipeline.py (CLI mode after user declines Power Mode)
  - app.py /update_mode/start (web UI, background thread)

Progress is reported via an optional callback:
    progress_cb(step_idx, total_steps, step_name, sub_message, sub_pct)
"""

import os
import sys
import subprocess
import traceback

TOTAL_STEPS = 7
SCRAPE_ONLY_STEPS = 4


def _noop_cb(step_idx, total_steps, step_name, sub_message="", sub_pct=0.0):
    pass


def _report(cb, idx, name, msg="", sub_pct=0.0, total=None):
    try:
        cb(idx, total or TOTAL_STEPS, name, msg, sub_pct)
    except Exception:
        traceback.print_exc()


def run_update_pipeline(progress_cb=None, test_mode=False, scrape_only=False):
    """
    Execute the full update pipeline.

    Args:
        progress_cb: optional callable(step_idx, total_steps, step_name, sub_message, sub_pct)
        test_mode: if True, skip slow scraping and train only 1 epoch (smoke test)
        scrape_only: if True, run steps 1-4 only (refresh data, no training).
                     Used by the daily cron on the CPU VM — training stays on the
                     on-demand GPU VM.

    Returns:
        dict with keys: status ("success"|"error"), message, steps (list of per-step results)
    """
    cb = progress_cb or _noop_cb
    steps_log = []
    total_steps_for_run = SCRAPE_ONLY_STEPS if scrape_only else TOTAL_STEPS

    def _report_tot(idx, name, msg="", sub_pct=0.0):
        _report(cb, idx, name, msg, sub_pct, total=total_steps_for_run)

    os.environ["FOBO_TEST_MODE"] = "true" if test_mode else "false"

    # Lazy imports so importing this module is cheap
    import scrape_flashscore
    import scrape_upcoming
    import check_data
    import storage_sync
    from league_urls import LEAGUE_URLS

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- STEP 1: Update historical match odds ---
    step = 1
    name = "Updating historical odds"
    _report_tot(step, name, "Starting...", 0.0)
    skip_historical = os.environ.get("FOBO_SKIP_HISTORICAL_UPDATE", "false").lower() == "true"
    try:
        script_path = os.path.join(script_dir, "old_matches", "update_odds_old_matches.py")
        if test_mode:
            _report_tot(step, name, "TEST MODE: skipped", 1.0)
            steps_log.append({"step": step, "name": name, "status": "skipped"})
        elif skip_historical:
            _report_tot(step, name, "SKIP: using existing old_matches/ CSVs", 1.0)
            steps_log.append({"step": step, "name": name, "status": "skipped"})
        elif os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], check=True, cwd=script_dir)
            _report_tot(step, name, "Completed", 1.0)
            steps_log.append({"step": step, "name": name, "status": "ok"})
        else:
            _report_tot(step, name, f"WARN: {script_path} not found", 1.0)
            steps_log.append({"step": step, "name": name, "status": "missing"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})
        # Non-fatal; continue

    # --- STEP 2: Scrape current-season results ---
    step = 2
    name = "Scraping match results"
    _report_tot(step, name, "Starting...", 0.0)
    try:
        if test_mode:
            _report_tot(step, name, "TEST MODE: skipped", 1.0)
            steps_log.append({"step": step, "name": name, "status": "skipped"})
        else:
            leagues = list(LEAGUE_URLS.items())
            total_leagues = len(leagues)
            for i, (league_name, url) in enumerate(leagues):
                _report_tot(step, name, f"Scraping {league_name}", i / max(total_leagues, 1))
                try:
                    scrape_flashscore.scrape_flashscore_final(url)
                except Exception as le:
                    print(f"[UPDATE] Failed to scrape {league_name}: {le}")
                    traceback.print_exc()
            _report_tot(step, name, f"Scraped {total_leagues} leagues", 1.0)
            steps_log.append({"step": step, "name": name, "status": "ok"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})

    # --- STEP 3: Scrape upcoming fixtures ---
    step = 3
    name = "Scraping upcoming fixtures"
    _report_tot(step, name, "Starting...", 0.0)
    try:
        if test_mode:
            _report_tot(step, name, "TEST MODE: skipped", 1.0)
            steps_log.append({"step": step, "name": name, "status": "skipped"})
        else:
            ok = scrape_upcoming.scrape_fixtures(days=30)
            msg = "Completed" if ok else "WARN: no matches returned"
            _report_tot(step, name, msg, 1.0)
            steps_log.append({"step": step, "name": name, "status": "ok" if ok else "warning"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})

    # --- STEP 4: Validate data integrity ---
    step = 4
    name = "Validating data"
    _report_tot(step, name, "Starting...", 0.0)
    try:
        check_data.run_checklist()
        _report_tot(step, name, "Completed", 1.0)
        steps_log.append({"step": step, "name": name, "status": "ok"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})

    # Short-circuit for scrape-only runs (the daily cron on the CPU VM)
    if scrape_only:
        if storage_sync.is_enabled():
            try:
                _report_tot(4, "Uploading CSVs to Cloud Storage", "data,history", 1.0)
                storage_sync.push_artifacts(["data", "history"])
            except Exception as e:
                print(f"[UPDATE] GCS push failed (non-fatal): {e}")
        return {
            "status": "success",
            "message": "Scrape-only update complete (training steps skipped).",
            "steps": steps_log,
        }

    # --- STEP 5: Retrain deep learning model ---
    step = 5
    name = "Training deep learning model"
    _report_tot(step, name, "Starting DL training...", 0.0)
    try:
        env = os.environ.copy()
        env["FOBO_TEST_MODE"] = "true" if test_mode else "false"
        dl_script = os.path.join(script_dir, "train_dl.py")
        subprocess.run([sys.executable, dl_script], check=True, cwd=script_dir, env=env)
        # Mark DL as trained so hybrid/app don't re-run
        os.environ["FOBO_SKIP_DL_TRAIN"] = "true"
        _report_tot(step, name, "Completed", 1.0)
        steps_log.append({"step": step, "name": name, "status": "ok"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})
        return {"status": "error", "message": f"DL training failed: {e}", "steps": steps_log}

    # --- STEP 6: Retrain hybrid XGBoost ensemble ---
    step = 6
    name = "Training hybrid (XGBoost) ensemble"
    _report_tot(step, name, "Starting hybrid training...", 0.0)
    try:
        import train_hybrid
        ok = train_hybrid.train_hybrid()
        msg = "Completed" if ok else "WARN: hybrid returned False"
        _report_tot(step, name, msg, 1.0)
        steps_log.append({"step": step, "name": name, "status": "ok" if ok else "warning"})
    except Exception as e:
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})
        # Non-fatal; continue to RL

    # --- STEP 7: Retrain PPO RL agent ---
    step = 7
    name = "Training RL (PPO) agent"
    _report_tot(step, name, "Starting PPO training...", 0.0)
    try:
        import torch
        import prediction_model as pm
        from prediction_model import LeagueAwareModel

        rl_state_dim = (pm.EMBED_DIM * 6) + pm.LEAGUE_EMBED_DIM + (pm.EMBED_DIM * 2)

        # Load freshly trained model to use as feature extractor
        model_path = os.path.join(script_dir, "models", "FOBO_LEAGUE_AWARE_current.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, "models", "FOBO_LEAGUE_AWARE_final.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained DL model found for RL bootstrapping")

        ckpt = torch.load(model_path, map_location=pm.DEVICE, weights_only=False)
        dl_model = LeagueAwareModel(
            ckpt["num_teams"], ckpt["num_leagues"],
            pm.EMBED_DIM, pm.LEAGUE_EMBED_DIM, pm.NUM_HEADS,
        ).to(pm.DEVICE)
        dl_model.load_state_dict(ckpt["model_state_dict"])
        dl_model.eval()

        ppo_epochs = 2 if test_mode else 20
        agent = pm.PPOAgent(state_dim=rl_state_dim, action_dim=4, lr=0.0003, entropy_coef=0.15).to(pm.DEVICE)
        agent = pm.train_ppo_agent(dl_model, agent, epochs=ppo_epochs)

        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(models_dir, "ppo_agent.pth"))
        _report_tot(step, name, "Completed", 1.0)
        steps_log.append({"step": step, "name": name, "status": "ok"})
    except Exception as e:
        traceback.print_exc()
        _report_tot(step, name, f"ERROR: {e}", 1.0)
        steps_log.append({"step": step, "name": name, "status": "error", "error": str(e)})

    # --- POST-STEP: sync artifacts to GCS if bucket configured ---
    if storage_sync.is_enabled():
        try:
            cats = ["data", "models", "history"]
            _report_tot(TOTAL_STEPS, "Uploading to Cloud Storage", ",".join(cats), 1.0)
            storage_sync.push_artifacts(cats)
        except Exception as e:
            print(f"[UPDATE] GCS push failed (non-fatal): {e}")

    return {"status": "success", "message": "Update pipeline complete", "steps": steps_log}
