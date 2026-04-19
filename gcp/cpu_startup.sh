#!/usr/bin/env bash
# CPU VM startup script — installs Docker, pulls the Flask image, runs it on port 80.
# Passed via `gcloud compute instances create ... --metadata-from-file=startup-script=gcp/cpu_startup.sh`
#
# Required metadata keys:
#   gcs-bucket      — shared bucket (e.g. "fobo-ai-shared")
#   repo-url        — https://github.com/jeonhs9110/epl.git

set -euo pipefail
exec > >(tee -a /var/log/fobo-cpu.log) 2>&1
echo "[cpu_startup] $(date -u +%FT%TZ) starting"

META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
BUCKET=$(curl -fsS -H "Metadata-Flavor: Google" "$META/gcs-bucket" || echo "")
REPO=$(curl -fsS -H "Metadata-Flavor: Google" "$META/repo-url" || echo "")

if [[ -z "$BUCKET" || -z "$REPO" ]]; then
  echo "[cpu_startup] ERROR: metadata gcs-bucket or repo-url missing"
  exit 1
fi

# 1. Install Docker
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg git
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io
  systemctl enable --now docker
fi

# 2. Checkout repo + build image
WORKDIR=/opt/fobo
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --depth=1 "$REPO" "$WORKDIR"
else
  git -C "$WORKDIR" fetch --depth=1 origin main
  git -C "$WORKDIR" reset --hard origin/main
fi
cd "$WORKDIR"

docker build -f gcp/Dockerfile.cpu -t fobo-cpu .

# 3. Stop any previous container then run fresh
docker rm -f fobo 2>/dev/null || true
ADMIN_TOKEN=$(curl -fsS -H "Metadata-Flavor: Google" "$META/admin-token" 2>/dev/null || echo "")

docker run -d \
  --name fobo \
  --restart unless-stopped \
  -p 80:8080 \
  -e FOBO_CLOUD=true \
  -e FOBO_GCS_BUCKET="$BUCKET" \
  -e FOBO_SKIP_TRAINING=true \
  -e FOBO_ADMIN_TOKEN="$ADMIN_TOKEN" \
  fobo-cpu

echo "[cpu_startup] $(date -u +%FT%TZ) container started"
docker logs --tail=20 fobo || true
