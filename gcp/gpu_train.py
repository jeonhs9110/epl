"""
GPU-VM training job: pull data -> train everything -> push models -> shut down.

Designed to be the one and only thing the GPU VM ever does. Flow:
  1. Pull CSVs + encoders from GCS bucket.
  2. Run full update pipeline (TEST_MODE=false, GPU will auto-detect).
  3. Push updated models + history back to bucket.
  4. Stop this Compute Engine instance (so you stop paying for GPU time).

Env vars expected:
  FOBO_GCS_BUCKET        — shared bucket (e.g. "fobo-ai-shared")
  FOBO_SHUTDOWN_ON_EXIT  — "true" to auto-stop the VM at the end (default true)
  FOBO_TEST_MODE         — "true" for a smoke test (1 epoch, no scraping)

Run manually:   python gcp/gpu_train.py
Via startup:    gcloud compute instances create ... --metadata-from-file=startup-script=gcp/gpu_train.sh
"""

import os
import sys
import time
import json
import queue
import threading
import traceback
import urllib.request

# Make sure we can import the project root when invoked as `python gcp/gpu_train.py`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _StreamingStdout:
    """
    Wraps sys.stdout so every print() also gets forwarded to the CPU VM's
    /admin/training_log endpoint in small batches. Lets the user watch
    Selenium worker output + per-match progress live in the Update modal.
    """

    def __init__(self, original, cpu_url, token, max_batch=30, flush_interval=1.2):
        self.original = original
        self.cpu_url = cpu_url.rstrip("/")
        self.token = token
        self.max_batch = max_batch
        self.flush_interval = flush_interval
        self._q = queue.Queue(maxsize=5000)
        self._buf = ""
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def write(self, text):
        try:
            self.original.write(text)
        except Exception:
            pass
        if not text:
            return len(text) if text else 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                try:
                    self._q.put_nowait(line)
                except queue.Full:
                    pass  # drop on overflow — training shouldn't block
        return len(text)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            lines = []
            deadline = time.time() + self.flush_interval
            while time.time() < deadline and len(lines) < self.max_batch:
                remaining = max(0.05, deadline - time.time())
                try:
                    lines.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            if lines:
                self._post(lines)

    def _post(self, lines):
        try:
            body = json.dumps({"lines": lines}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.cpu_url}/admin/training_log",
                data=body, method="POST",
                headers={"X-Admin-Token": self.token, "Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=4).read()
        except Exception:
            pass  # streaming is best-effort; never break training


def _enable_log_streaming():
    cpu_url = os.environ.get("FOBO_CPU_URL", "").strip()
    token = os.environ.get("FOBO_ADMIN_TOKEN", "").strip()
    if not cpu_url or not token:
        return
    sys.stdout = _StreamingStdout(sys.stdout, cpu_url, token)
    sys.stderr = _StreamingStdout(sys.stderr, cpu_url, token)
    print("[GPU_TRAIN] stdout streaming to CPU VM enabled.")


def _metadata_get(path):
    """Fetch a value from the GCE metadata server (no auth needed from inside the VM)."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"http://metadata.google.internal/computeMetadata/v1/{path}",
            headers={"Metadata-Flavor": "Google"},
        )
        return urllib.request.urlopen(req, timeout=3).read().decode().strip()
    except Exception:
        return ""


def shutdown_vm():
    """Self-DELETE (not stop). User wants the GPU VM gone after training so there's
    no lingering disk cost. Resolves instance name + zone from the GCE metadata
    server (HOSTNAME is not reliably populated in every shell)."""
    import subprocess
    print("\n[GPU_TRAIN] Deleting self VM (training complete)...")

    instance = _metadata_get("instance/name") or os.environ.get("HOSTNAME", "")
    zone_path = _metadata_get("instance/zone")
    zone = zone_path.split("/")[-1] if zone_path else os.environ.get("FOBO_GCP_ZONE", "us-central1-a")

    print(f"[GPU_TRAIN]   instance={instance!r} zone={zone!r}")

    # Preferred: gcloud delete (works on the DL VM image which has gcloud built in)
    if instance:
        try:
            subprocess.run(
                ["gcloud", "compute", "instances", "delete", instance,
                 "--zone", zone, "--quiet"],
                check=False,
                timeout=120,
            )
            return
        except Exception as e:
            print(f"[GPU_TRAIN] gcloud delete failed: {e}; trying REST API fallback")

    # Fallback #1: Compute Engine REST API via the VM's metadata token
    try:
        import urllib.request, json as _json
        token_req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(token_req, timeout=5) as r:
            token = _json.loads(r.read().decode())["access_token"]
        project = _metadata_get("project/project-id")
        del_req = urllib.request.Request(
            f"https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instances/{instance}",
            method="DELETE",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(del_req, timeout=30) as r:
            print(f"[GPU_TRAIN] REST delete ok: {r.status}")
            return
    except Exception as e:
        print(f"[GPU_TRAIN] REST delete failed: {e}; last resort shutdown -h now")

    # Fallback #2: pull the plug. VM stays as TERMINATED but at least stops billing.
    try:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False, timeout=30)
    except Exception as e2:
        print(f"[GPU_TRAIN] shutdown failed: {e2}")


def _ppo_only_train():
    """
    Re-train only the PPO agent against the DL model already sitting in
    models/FOBO_LEAGUE_AWARE_current.pth. Used when a full training run
    succeeded through DL+hybrid but step 7 (PPO) failed — we don't want
    to redo 25 min of DL training just to fix a 3 MB checkpoint.

    Mirrors update_pipeline.run_update_pipeline step 7 logic, including
    the encoders-based num_teams/num_leagues recovery path.
    """
    import pickle
    import torch
    import prediction_model as pm
    from prediction_model import LeagueAwareModel

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    models_dir = os.path.join(project_root, "models")

    model_path = os.path.join(models_dir, "FOBO_LEAGUE_AWARE_current.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "FOBO_LEAGUE_AWARE_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained DL model found for PPO bootstrap")

    ckpt = torch.load(model_path, map_location=pm.DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "num_teams" in ckpt and "model_state_dict" in ckpt:
        num_teams = ckpt["num_teams"]
        num_leagues = ckpt["num_leagues"]
        state_dict = ckpt["model_state_dict"]
    else:
        encoders_path = os.path.join(project_root, "encoders.pkl")
        with open(encoders_path, "rb") as _f:
            le = pickle.load(_f)
        num_teams = len(le["le_team"].classes_)
        num_leagues = len(le["le_league"].classes_)
        state_dict = ckpt

    # LeagueAwareModel takes only (num_teams, num_leagues, dataset_adj=None)
    dl_model = LeagueAwareModel(num_teams, num_leagues).to(pm.DEVICE)
    dl_model.load_state_dict(state_dict)
    dl_model.eval()

    rl_state_dim = (pm.EMBED_DIM * 6) + pm.LEAGUE_EMBED_DIM + (pm.EMBED_DIM * 2)
    agent = pm.PPOAgent(state_dim=rl_state_dim, action_dim=4, lr=0.0003, entropy_coef=0.15).to(pm.DEVICE)

    ppo_epochs = 2 if os.environ.get("FOBO_TEST_MODE", "false").lower() == "true" else 20
    print(f"[GPU_TRAIN] PPO-only: training for {ppo_epochs} epochs on calibrated-DL features...")
    agent = pm.train_ppo_agent(dl_model, agent, epochs=ppo_epochs)

    os.makedirs(models_dir, exist_ok=True)
    ppo_path = os.path.join(models_dir, "ppo_agent.pth")
    torch.save(agent.state_dict(), ppo_path)
    sz = os.path.getsize(ppo_path)
    print(f"[GPU_TRAIN] PPO agent saved to {ppo_path} ({sz/1024/1024:.1f} MB)")


def main():
    # Hook stdout/stderr to the CPU VM's /admin/training_log FIRST so every
    # subsequent print — including Selenium worker output + per-match log
    # lines from inside scrape_flashscore — shows up in the Update modal.
    _enable_log_streaming()

    print("=" * 60)
    print("FOBO AI — GPU Training Job")
    print("=" * 60)

    import storage_sync
    import update_pipeline

    if not storage_sync.is_enabled():
        print("[GPU_TRAIN] FOBO_GCS_BUCKET not set — will use local files only.")
    else:
        print("[GPU_TRAIN] Pulling latest data + encoders from bucket...")
        storage_sync.pull_artifacts(["data", "encoders", "history"])

    test_mode = os.environ.get("FOBO_TEST_MODE", "false").lower() == "true"
    print(f"[GPU_TRAIN] Running update pipeline (test_mode={test_mode})...")

    cpu_url = os.environ.get("FOBO_CPU_URL", "").strip()
    admin_token = os.environ.get("FOBO_ADMIN_TOKEN", "").strip()

    def post_progress(step_idx, total_steps, step_name, sub_message, sub_pct, status="running"):
        if not cpu_url or not admin_token:
            return
        try:
            import urllib.request, json as _json
            body = _json.dumps({
                "step": step_idx, "total_steps": total_steps,
                "step_name": step_name, "sub_message": sub_message,
                "sub_pct": sub_pct, "status": status,
            }).encode()
            req = urllib.request.Request(
                f"{cpu_url}/admin/training_progress",
                data=body, method="POST",
                headers={"X-Admin-Token": admin_token, "Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            # Non-fatal; keep training
            print(f"[GPU_TRAIN] progress post failed: {e}")

    def cli_cb(step_idx, total_steps, step_name, sub_message="", sub_pct=0.0):
        overall = int(100 * (step_idx - 1 + sub_pct) / total_steps)
        print(f"  [{overall:3d}%] Step {step_idx}/{total_steps}: {step_name} — {sub_message}")
        post_progress(step_idx, total_steps, step_name, sub_message, sub_pct, status="running")

    # If the CPU VM has already scraped (the typical flow), skip steps 1-3 on the
    # GPU VM to save 30-40 minutes. Controlled by FOBO_SKIP_SCRAPE env var.
    skip_scrape = os.environ.get("FOBO_SKIP_SCRAPE", "false").lower() == "true"
    ppo_only = os.environ.get("FOBO_PPO_ONLY", "false").lower() == "true"
    print(f"[GPU_TRAIN] skip_scrape={skip_scrape}  ppo_only={ppo_only}")

    try:
        # Tell the CPU VM a training run is starting
        post_progress(0, 7, "Starting GPU training", "Pulling data from bucket...", 0.0, status="running")

        if ppo_only:
            # Shortcut: re-train ONLY the PPO agent using the fresh DL/hybrid
            # models already in the bucket. Used to recover from a failed
            # step 7 without redoing DL (~25 min) or hybrid (~5 min).
            print("[GPU_TRAIN] PPO-only mode: pulling models from bucket, training PPO, pushing back.")
            if storage_sync.is_enabled():
                storage_sync.pull_artifacts(["models", "encoders"])
            _ppo_only_train()
            if storage_sync.is_enabled():
                print("[GPU_TRAIN] Pushing fresh ppo_agent.pth to bucket...")
                storage_sync.push_artifacts(["models"])
            post_progress(7, 7, "PPO-only training complete", "PPO agent pushed to bucket", 1.0, status="complete")
            result = {"status": "success", "message": "PPO-only run complete", "steps": []}
        else:
            result = update_pipeline.run_update_pipeline(
                progress_cb=cli_cb,
                test_mode=test_mode,
                skip_scrape=skip_scrape,
            )
        print(f"\n[GPU_TRAIN] Pipeline {result['status']}: {result['message']}")
        for s in result["steps"]:
            print(f"  [{s['status']}] {s['name']}")
    except Exception as e:
        traceback.print_exc()
        print(f"[GPU_TRAIN] FATAL: {e}")
        post_progress(0, 7, "GPU training failed", str(e), 1.0, status="error")
        # Still try to push whatever we have
        if storage_sync.is_enabled():
            storage_sync.push_artifacts(["models", "history"])
        if os.environ.get("FOBO_SHUTDOWN_ON_EXIT", "true").lower() == "true":
            shutdown_vm()
        sys.exit(1)

    # update_pipeline.py already pushes at the end when bucket is set,
    # but push again to be safe. Include "data" so any newly-scraped CSVs
    # definitely land in the bucket before the GPU VM shuts down — otherwise
    # the CPU VM would have no record of the new matches.
    if storage_sync.is_enabled():
        print("[GPU_TRAIN] Pushing data + models + history to bucket...")
        storage_sync.push_artifacts(["data", "models", "history"])

    # Tell the CPU VM training is done
    post_progress(7, 7, "Training complete", "Models uploaded to bucket", 1.0, status="complete")

    # Tell the CPU VM to pull fresh models immediately (backup + live reload).
    # Best-effort; CPU VM also polls the bucket every 15 min as a fallback.
    cpu_host = os.environ.get("FOBO_CPU_URL", "").strip()
    admin_token = os.environ.get("FOBO_ADMIN_TOKEN", "").strip()
    if cpu_host and admin_token:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{cpu_host}/admin/reload_models",
                method="POST",
                headers={"X-Admin-Token": admin_token},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                print(f"[GPU_TRAIN] CPU reload response: {resp.status} {resp.read()[:200]}")
        except Exception as e:
            print(f"[GPU_TRAIN] CPU reload call failed (non-fatal, 15-min poll will catch up): {e}")

    if os.environ.get("FOBO_SHUTDOWN_ON_EXIT", "true").lower() == "true":
        shutdown_vm()


if __name__ == "__main__":
    main()
