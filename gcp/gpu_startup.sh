#!/usr/bin/env bash
# GPU VM startup script — passed via --metadata-from-file=startup-script=...
# Installs deps, clones repo, runs training, uploads models, stops the VM.
#
# Required metadata keys (set via --metadata=KEY=VALUE at `gcloud compute instances create`):
#   gcs-bucket       — shared bucket name (e.g. "fobo-ai-shared")
#   repo-url         — https://github.com/jeonhs9110/epl.git
#   gcp-zone         — zone of this VM (e.g. us-central1-a) so shutdown works
#   test-mode        — "true" to smoke-test instead of full train (optional)

set -euo pipefail
exec > >(tee -a /var/log/fobo-gpu.log) 2>&1
echo "[gpu_startup] $(date -u +%FT%TZ) starting"

# 1. Pull metadata
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
BUCKET=$(curl -fsS -H "Metadata-Flavor: Google" "$META/gcs-bucket" || echo "")
REPO=$(curl -fsS -H "Metadata-Flavor: Google" "$META/repo-url" || echo "")
ZONE=$(curl -fsS -H "Metadata-Flavor: Google" "$META/gcp-zone" || echo "us-central1-a")
TEST_MODE=$(curl -fsS -H "Metadata-Flavor: Google" "$META/test-mode" || echo "false")

if [[ -z "$BUCKET" || -z "$REPO" ]]; then
  echo "[gpu_startup] ERROR: metadata gcs-bucket or repo-url missing"
  exit 1
fi

# 2. System deps (Deep Learning VM images have Python + CUDA preinstalled)
apt-get update -y
apt-get install -y --no-install-recommends git python3-pip

# 3. Checkout repo
WORKDIR=/opt/fobo
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --depth=1 "$REPO" "$WORKDIR"
else
  git -C "$WORKDIR" fetch --depth=1 origin main
  git -C "$WORKDIR" reset --hard origin/main
fi
cd "$WORKDIR"

# 4. Python deps (CUDA-enabled torch from the DL VM image, keep whatever is installed)
pip3 install --upgrade pip
pip3 install -r requirements.txt
# If the base image doesn't already have CUDA torch, install the CUDA wheel:
python3 -c "import torch; print(torch.cuda.is_available())" \
  || pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Run the training job (pushes to bucket + self-shuts-down at the end)
export FOBO_CLOUD=true
export FOBO_GCS_BUCKET="$BUCKET"
export FOBO_GCP_ZONE="$ZONE"
export FOBO_TEST_MODE="$TEST_MODE"
export FOBO_SHUTDOWN_ON_EXIT=true

python3 gcp/gpu_train.py

echo "[gpu_startup] $(date -u +%FT%TZ) done"
