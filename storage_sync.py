"""
Google Cloud Storage sync for CSVs + model weights.

Two VMs share state through a single GCS bucket:
  - CPU VM (24/7) writes freshly scraped CSVs and reads latest model weights.
  - GPU VM (on-demand) reads CSVs to train, writes updated model weights.

Layout:
    gs://$FOBO_GCS_BUCKET/
        data/*.csv           # FOOTBALL_*.csv + UPCOMING_MATCHES.csv
        models/*.pth,*.json  # model weights + hybrid model
        history/*.json       # training_history, rl_history, etc.

Env vars:
    FOBO_GCS_BUCKET — bucket name (e.g. "fobo-ai-shared"). If unset, all
                      sync calls are no-ops so local dev is unaffected.

Usage:
    from storage_sync import push_artifacts, pull_artifacts
    pull_artifacts(["data", "models"])   # GPU VM before training
    push_artifacts(["models", "history"]) # GPU VM after training
    push_artifacts(["data"])              # CPU VM after scraping
"""

import os
import glob
import traceback

# File categories -> (local_dir_relative, gcs_prefix, glob_patterns)
CATEGORIES = {
    "data": (".", "data", ["FOOTBALL_*.csv", "UPCOMING_MATCHES.csv"]),
    "models": ("models", "models", ["*.pth", "*.json", "*.joblib"]),
    "history": (".", "history", ["training_history.json", "bet_history.json", "optimal_seq_lengths.json"]),
    "encoders": (".", "encoders", ["encoders.pkl"]),
}


def _bucket_name():
    return os.environ.get("FOBO_GCS_BUCKET", "").strip()


def _get_bucket():
    """Return a GCS bucket handle, or None if disabled."""
    name = _bucket_name()
    if not name:
        return None
    try:
        from google.cloud import storage
    except ImportError:
        print("[storage_sync] google-cloud-storage not installed; skipping sync")
        return None
    try:
        client = storage.Client()
        return client.bucket(name)
    except Exception as e:
        print(f"[storage_sync] failed to open bucket {name!r}: {e}")
        return None


def is_enabled():
    return bool(_bucket_name())


def push_artifacts(categories):
    """Upload local files matching the given categories to the bucket."""
    bucket = _get_bucket()
    if bucket is None:
        return {"status": "skipped", "reason": "no bucket configured"}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploaded = []
    for cat in categories:
        if cat not in CATEGORIES:
            print(f"[storage_sync] unknown category: {cat}")
            continue
        local_dir, gcs_prefix, patterns = CATEGORIES[cat]
        base = os.path.join(script_dir, local_dir)
        for pat in patterns:
            for path in glob.glob(os.path.join(base, pat)):
                name = os.path.basename(path)
                blob = bucket.blob(f"{gcs_prefix}/{name}")
                try:
                    blob.upload_from_filename(path)
                    uploaded.append(f"{gcs_prefix}/{name}")
                    print(f"[storage_sync] pushed {gcs_prefix}/{name}")
                except Exception as e:
                    print(f"[storage_sync] failed to push {path}: {e}")
                    traceback.print_exc()

    return {"status": "ok", "uploaded": uploaded}


def pull_artifacts(categories):
    """Download bucket objects matching the given categories into local dirs."""
    bucket = _get_bucket()
    if bucket is None:
        return {"status": "skipped", "reason": "no bucket configured"}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloaded = []
    for cat in categories:
        if cat not in CATEGORIES:
            print(f"[storage_sync] unknown category: {cat}")
            continue
        local_dir, gcs_prefix, _ = CATEGORIES[cat]
        base = os.path.join(script_dir, local_dir)
        os.makedirs(base, exist_ok=True)
        for blob in bucket.list_blobs(prefix=f"{gcs_prefix}/"):
            name = blob.name.split("/", 1)[1] if "/" in blob.name else blob.name
            if not name:
                continue
            dest = os.path.join(base, name)
            try:
                blob.download_to_filename(dest)
                downloaded.append(f"{gcs_prefix}/{name}")
                print(f"[storage_sync] pulled {gcs_prefix}/{name}")
            except Exception as e:
                print(f"[storage_sync] failed to pull {blob.name}: {e}")
                traceback.print_exc()

    return {"status": "ok", "downloaded": downloaded}
