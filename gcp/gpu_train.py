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
import traceback

# Make sure we can import the project root when invoked as `python gcp/gpu_train.py`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def shutdown_vm():
    """Best-effort self-shutdown via the Compute Engine metadata server."""
    import subprocess
    print("\n[GPU_TRAIN] Shutting down VM...")
    try:
        # Preferred: let the metadata service resolve the instance and zone
        subprocess.run(
            ["gcloud", "compute", "instances", "stop", os.environ.get("HOSTNAME", "self"),
             "--zone", os.environ.get("FOBO_GCP_ZONE", "us-central1-a")],
            check=False,
            timeout=60,
        )
    except Exception as e:
        print(f"[GPU_TRAIN] gcloud shutdown failed: {e}; falling back to `sudo shutdown`")
        try:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False, timeout=30)
        except Exception as e2:
            print(f"[GPU_TRAIN] shutdown failed: {e2}")


def main():
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

    try:
        # Tell the CPU VM a training run is starting
        post_progress(0, 7, "Starting GPU training", "Pulling data from bucket...", 0.0, status="running")
        result = update_pipeline.run_update_pipeline(progress_cb=cli_cb, test_mode=test_mode)
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
