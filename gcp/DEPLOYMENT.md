# FOBO AI — Google Cloud Deployment Guide

Target architecture (budget: < $150/month):

```
                                     ┌──────────────────────────┐
 visitors (24/7) ──── http://IP ───▶ │  CPU VM (e2-small)       │
                                     │  Flask + Selenium+xvfb   │
                                     │  ~$13–24/month           │
                                     └──────────┬───────────────┘
                                                │
                                     ┌──────────▼───────────────┐
                                     │  Cloud Storage bucket    │
                                     │  data/ models/ history/  │
                                     │  ~$1/month               │
                                     └──────────▲───────────────┘
                                                │ (on-demand)
                                     ┌──────────┴───────────────┐
                                     │  GPU VM (n1-standard-4   │
                                     │  + T4). Trains, pushes   │
                                     │  models, auto-shuts-down │
                                     │  ~$0.35/hr × ~10 hr/mo   │
                                     └──────────────────────────┘
```

**Total cost target:** ≈ $20–40/month for normal use.

---

## Prerequisites

1. Install `gcloud` locally: https://cloud.google.com/sdk/docs/install
2. Log in:
   ```bash
   gcloud auth login
   ```
3. Your GitHub repo URL (default assumed): `https://github.com/jeonhs9110/epl.git`

---

## One-time setup

All following commands set variables once then reuse them. Run from the
project root on your local machine.

### 1. Pick a project ID and region

```bash
# IMPORTANT: must be a brand new project, separate from "AI-influencer"
export PROJECT_ID="fobo-ai-$(whoami)-$(date +%s | tail -c 5)"
export REGION="us-central1"
export ZONE="us-central1-a"
export BUCKET="${PROJECT_ID}-shared"
export REPO_URL="https://github.com/jeonhs9110/epl.git"
```

### 2. Create the project, link billing, enable APIs

```bash
# Create project
gcloud projects create "$PROJECT_ID" --name="FOBO AI"
gcloud config set project "$PROJECT_ID"

# Link to billing (list then link)
gcloud beta billing accounts list
export BILLING_ACCOUNT=<paste the account ID from above>
gcloud beta billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"

# Enable required services
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    iamcredentials.googleapis.com
```

### 3. Set a monthly budget alert (prevents surprises)

```bash
gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT" \
    --display-name="FOBO AI monthly cap" \
    --budget-amount=150USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100
```

### 4. Create the shared Cloud Storage bucket

```bash
gcloud storage buckets create "gs://$BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access
```

### 5. Create a service account the VMs will use

```bash
gcloud iam service-accounts create fobo-vm \
    --display-name="FOBO VM service account"

export SA_EMAIL="fobo-vm@${PROJECT_ID}.iam.gserviceaccount.com"

# Read/write to the bucket
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.objectAdmin"

# Let GPU VM stop itself
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/compute.instanceAdmin.v1"
```

---

## A. Deploy the CPU VM (always-on frontend)

This box serves the website 24/7 and runs Selenium scraping.

```bash
gcloud compute instances create fobo-cpu \
    --zone="$ZONE" \
    --machine-type=e2-small \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --service-account="$SA_EMAIL" \
    --scopes=cloud-platform \
    --tags=http-server \
    --metadata="gcs-bucket=$BUCKET,repo-url=$REPO_URL" \
    --metadata-from-file=startup-script=gcp/cpu_startup.sh
```

Open port 80:

```bash
gcloud compute firewall-rules create allow-http \
    --allow=tcp:80 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server
```

Get the public IP (wait ~3–5 min for Docker build to finish first):

```bash
gcloud compute instances describe fobo-cpu --zone="$ZONE" \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
```

Visit `http://<IP>` — the app should respond. First boot takes ~5–7 min
while Docker installs and Chrome is built.

**Cost so far:** ~$13/month for e2-small + ~$1/month bucket.

---

## B. Do the first training run on a GPU VM

You need models in the bucket before the CPU VM can predict properly.
Spin up a GPU VM. It will train, push models to the bucket, and stop itself.

```bash
gcloud compute instances create fobo-gpu \
    --zone="$ZONE" \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --service-account="$SA_EMAIL" \
    --scopes=cloud-platform \
    --metadata="gcs-bucket=$BUCKET,repo-url=$REPO_URL,gcp-zone=$ZONE,test-mode=false,install-nvidia-driver=True" \
    --metadata-from-file=startup-script=gcp/gpu_startup.sh
```

The VM boots, clones the repo, trains everything, uploads models,
and calls `gcloud compute instances stop fobo-gpu` on itself.

Watch progress:

```bash
gcloud compute instances tail-serial-port-output fobo-gpu --zone="$ZONE"
```

When the instance is in `TERMINATED` state, training is done and models
are in `gs://$BUCKET/models/`.

**Restart the CPU VM** so it picks up the freshly trained models:

```bash
gcloud compute instances reset fobo-cpu --zone="$ZONE"
```

---

## Retraining later

After a few weeks of new match data you can retrain. The GPU VM is
`TERMINATED`, so starting it costs nothing until boot.

```bash
# Quickest: just start the stopped instance. It re-runs the startup script.
gcloud compute instances start fobo-gpu --zone="$ZONE"
```

If you want to change test-mode or other params, delete + recreate with
the same command from section B.

---

## Smoke test (recommended before your first full train)

To verify the wiring without waiting 30+ min for a full train:

```bash
# Recreate the GPU VM in test mode (1 epoch, skip scraping)
gcloud compute instances delete fobo-gpu --zone="$ZONE" --quiet
gcloud compute instances create fobo-gpu \
    --zone="$ZONE" --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121 --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --service-account="$SA_EMAIL" --scopes=cloud-platform \
    --metadata="gcs-bucket=$BUCKET,repo-url=$REPO_URL,gcp-zone=$ZONE,test-mode=true,install-nvidia-driver=True" \
    --metadata-from-file=startup-script=gcp/gpu_startup.sh
```

This finishes in ~5 min and verifies that:
- Bucket permissions work (pull data → push models)
- Chrome/Selenium install isn't needed on GPU side (skipped in test mode)
- Training code runs on GPU
- Self-shutdown works

The CPU VM has the same "Test Mode" toggle in the in-app Update modal
for verifying the frontend flow without long training.

---

## Useful operational commands

```bash
# SSH into CPU VM for debugging
gcloud compute ssh fobo-cpu --zone="$ZONE"

# View Flask container logs
gcloud compute ssh fobo-cpu --zone="$ZONE" --command="docker logs --tail=200 fobo"

# Restart the Flask container after code updates
gcloud compute ssh fobo-cpu --zone="$ZONE" --command="sudo /var/run/google.startup.d/..."
# Or simpler: reset the VM to re-run the full startup script
gcloud compute instances reset fobo-cpu --zone="$ZONE"

# List bucket contents
gcloud storage ls -r "gs://$BUCKET/**"

# Teardown everything (when you're done experimenting)
gcloud compute instances delete fobo-cpu fobo-gpu --zone="$ZONE" --quiet
gcloud storage rm -r "gs://$BUCKET"
gcloud projects delete "$PROJECT_ID"
```

---

## Cost guardrails (review monthly)

- CPU VM `e2-small` running 24/7: ~$13/month
- CPU VM `e2-medium` if e2-small feels slow: ~$24/month
- GPU VM `n1-standard-4 + T4`: ~$0.35/hr — **make sure it says TERMINATED
  when idle**. If it says RUNNING and you're not training, `gcloud compute
  instances stop fobo-gpu` immediately.
- Bucket: ~$0.02/GB/month. Models+CSVs ≈ 50 MB → pennies.
- Egress (visitor traffic): first 1 GB/month free, then $0.12/GB.

The budget alert from step 3 will email you at 50/90/100% of $150.
