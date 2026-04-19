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

# 2. System deps (Deep Learning VM images have Python + CUDA preinstalled).
# Chrome + Xvfb are needed when running the full (non-test) pipeline, because
# steps 1-3 use Selenium against Flashscore (which blocks --headless Chrome,
# so we run real Chrome inside a virtual framebuffer).
apt-get update -y
apt-get install -y --no-install-recommends \
  git python3-pip wget gnupg ca-certificates curl unzip \
  xvfb \
  fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 \
  libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 \
  libx11-xcb1 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 xdg-utils

# Install Google Chrome stable (chromedriver comes from webdriver-manager at runtime)
if ! command -v google-chrome >/dev/null 2>&1; then
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get install -y /tmp/chrome.deb
  rm -f /tmp/chrome.deb
fi

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
# --ignore-installed avoids Ubuntu 22.04 distutils-packages (e.g. blinker) breaking
# the install when pip tries to uninstall them.
pip3 install --upgrade pip
pip3 install --ignore-installed --no-warn-conflicts -r requirements.txt
# If the base image doesn't already have CUDA torch, install the CUDA wheel:
python3 -c "import torch; print(torch.cuda.is_available())" \
  || pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Run the training job (pushes to bucket + self-shuts-down at the end)
export FOBO_CLOUD=true
export FOBO_GCS_BUCKET="$BUCKET"
export FOBO_GCP_ZONE="$ZONE"
export FOBO_TEST_MODE="$TEST_MODE"
export FOBO_SHUTDOWN_ON_EXIT=true
# Force XGBoost/LightGBM to CPU — cuda mode on DL Platform common-cu129
# segfaults inside libc. DL training (PyTorch) still runs on GPU.
export FOBO_XGB_CPU=true

# Optional: tell the CPU VM to reload its models as soon as training finishes.
export FOBO_CPU_URL=$(curl -fsS -H "Metadata-Flavor: Google" "$META/cpu-url" || echo "")
export FOBO_ADMIN_TOKEN=$(curl -fsS -H "Metadata-Flavor: Google" "$META/admin-token" || echo "")

# Start Xvfb on :99 so Selenium's Chrome has a display (Flashscore blocks true headless).
Xvfb :99 -screen 0 1920x1080x24 -ac +extension RANDR +render -noreset &
export DISPLAY=:99
sleep 1

python3 gcp/gpu_train.py

echo "[gpu_startup] $(date -u +%FT%TZ) done"
