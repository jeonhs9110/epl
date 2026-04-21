"""
Spawn / delete the GPU training VM from inside the CPU VM.

The CPU VM's service account (fobo-vm@...) has roles/compute.instanceAdmin.v1,
so it can create, describe, and delete VMs via the Compute Engine REST API.
We authenticate with the VM's own metadata-server access token — no key files.

Used by:
  - /update_mode/start on the CPU VM, when the user kicks off a full retrain:
      1. scrape fresh match data (step 1-4 of update_pipeline)
      2. call spawn_gpu_vm() → L4 Spot VM in asia-northeast3-b
      3. GPU VM runs gpu_startup.sh, which calls gpu_train.py
      4. When gpu_train.py finishes, it calls delete_self_vm() to self-destruct
"""

import os
import json
import time
import urllib.request
import urllib.error


PROJECT = os.environ.get("GCP_PROJECT", "fobo-ai-jeonhs9110")
DEFAULT_ZONE = os.environ.get("GCP_ZONE", "asia-northeast3-b")
REPO_URL = os.environ.get("FOBO_REPO_URL", "https://github.com/jeonhs9110/epl.git")


def _access_token():
    """Fetch a short-lived OAuth access token from the VM metadata server."""
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode())["access_token"]


def _ce_request(method, path, body=None, timeout=30):
    token = _access_token()
    url = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}


def get_instance(name, zone=DEFAULT_ZONE):
    return _ce_request("GET", f"/zones/{zone}/instances/{name}")


def list_instances(zone=DEFAULT_ZONE):
    return _ce_request("GET", f"/zones/{zone}/instances")


def delete_instance(name, zone=DEFAULT_ZONE):
    return _ce_request("DELETE", f"/zones/{zone}/instances/{name}")


def delete_self_vm():
    """Called from inside the GPU VM after training is done. Deletes itself."""
    name = _metadata_get("instance/name")
    zone_path = _metadata_get("instance/zone")
    zone = zone_path.split("/")[-1] if zone_path else DEFAULT_ZONE
    return delete_instance(name, zone)


def _metadata_get(path):
    req = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        return resp.read().decode().strip()


def _read_startup_script():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "gcp", "gpu_startup.sh"), "r", encoding="utf-8") as f:
        return f.read()


def build_gpu_instance_body(
    name,
    zone,
    bucket_name,
    admin_token,
    cpu_url,
    accelerator="nvidia-l4",
    machine_type="g2-standard-4",
    test_mode=False,
    service_account_email=None,
):
    """Build the Compute Engine Instance body for the GPU training VM."""
    if service_account_email is None:
        # Use the same SA as the CPU VM
        service_account_email = f"fobo-vm@{PROJECT}.iam.gserviceaccount.com"

    image = (
        f"projects/deeplearning-platform-release/global/images/family/"
        "common-cu129-ubuntu-2204-nvidia-580"
    )

    return {
        "name": name,
        "machineType": f"zones/{zone}/machineTypes/{machine_type}",
        "disks": [{
            "boot": True,
            "autoDelete": True,
            "initializeParams": {
                "sourceImage": image,
                "diskSizeGb": "50",
                "diskType": f"zones/{zone}/diskTypes/pd-balanced",
            },
        }],
        "networkInterfaces": [{
            "network": "global/networks/default",
            "accessConfigs": [{
                "type": "ONE_TO_ONE_NAT",
                "name": "External NAT",
            }],
        }],
        "guestAccelerators": [{
            "acceleratorType": f"zones/{zone}/acceleratorTypes/{accelerator}",
            "acceleratorCount": 1,
        }],
        "scheduling": {
            "onHostMaintenance": "TERMINATE",
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
        },
        "serviceAccounts": [{
            "email": service_account_email,
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        }],
        "metadata": {
            "items": [
                {"key": "gcs-bucket", "value": bucket_name},
                {"key": "repo-url", "value": REPO_URL},
                {"key": "gcp-zone", "value": zone},
                {"key": "test-mode", "value": "true" if test_mode else "false"},
                {"key": "install-nvidia-driver", "value": "True"},
                {"key": "admin-token", "value": admin_token},
                {"key": "cpu-url", "value": cpu_url},
                {"key": "startup-script", "value": _read_startup_script()},
            ],
        },
    }


def spawn_gpu_vm(
    name="fobo-gpu",
    bucket_name=None,
    admin_token=None,
    cpu_url=None,
    test_mode=False,
    zones_to_try=None,
    accelerators_to_try=None,
):
    """
    Create the GPU training VM, trying multiple zone/accelerator combinations
    if Spot capacity is unavailable.

    Returns dict: {"status": "ok"|"error", "instance": ..., "zone": ..., "message": ...}
    """
    bucket_name = bucket_name or os.environ.get("FOBO_GCS_BUCKET", f"{PROJECT}-shared")
    admin_token = admin_token or os.environ.get("FOBO_ADMIN_TOKEN", "")
    cpu_url = cpu_url or os.environ.get("FOBO_CPU_URL") or "http://34.64.147.124"

    if not admin_token:
        return {"status": "error", "message": "FOBO_ADMIN_TOKEN not set on CPU VM."}

    zones_to_try = zones_to_try or [
        "asia-northeast3-b",
        "asia-northeast3-a",
        "asia-northeast3-c",
    ]
    # (accelerator, machine_type) pairs
    accelerators_to_try = accelerators_to_try or [
        ("nvidia-l4", "g2-standard-4"),
        ("nvidia-tesla-t4", "n1-standard-4"),
    ]

    # If the VM already exists, return its details rather than failing
    existing = get_instance(name, zones_to_try[0])
    if existing[0] == 200:
        return {
            "status": "ok",
            "instance": existing[1],
            "zone": zones_to_try[0],
            "message": "Instance already exists",
        }

    last_err = None
    for zone in zones_to_try:
        for accel, mt in accelerators_to_try:
            body = build_gpu_instance_body(
                name=name, zone=zone,
                bucket_name=bucket_name,
                admin_token=admin_token, cpu_url=cpu_url,
                accelerator=accel, machine_type=mt,
                test_mode=test_mode,
            )
            status, resp = _ce_request(
                "POST", f"/zones/{zone}/instances", body=body, timeout=60
            )
            if status in (200, 201):
                return {
                    "status": "ok",
                    "zone": zone,
                    "accelerator": accel,
                    "machine_type": mt,
                    "operation": resp,
                    "message": f"Created {name} with {accel} in {zone}",
                }
            last_err = resp
            # Quota / stockout / capacity errors → try next combo
            err_str = json.dumps(resp)
            if any(s in err_str for s in ["STOCKOUT", "resource_availability", "Quota"]):
                continue
            # Other errors → bail
            return {"status": "error", "zone": zone, "message": err_str}

    return {
        "status": "error",
        "message": f"All zone/accelerator combos exhausted. Last error: {last_err}",
    }


def wait_for_vm_gone(name="fobo-gpu", zone=DEFAULT_ZONE, timeout_s=7200):
    """Poll until the GPU VM is deleted (or timeout). Returns True if gone."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, _resp = get_instance(name, zone)
        if status == 404:
            return True
        time.sleep(20)
    return False


if __name__ == "__main__":
    # Quick CLI smoke test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "spawn":
        print(json.dumps(spawn_gpu_vm(test_mode=True), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "delete":
        print(json.dumps(delete_instance("fobo-gpu"), indent=2))
    else:
        print(json.dumps(list_instances(), indent=2))
