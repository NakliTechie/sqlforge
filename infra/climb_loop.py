"""Relaunch loop for climb 1 on GCP spot (runs detached on the Mac; outlives the Claude session).

Every POLL seconds:
  DONE marker in the bucket            → alert, exit 0
  STOP marker (manual or written here) → delete climb VMs, alert, exit 1
  spend ≥ CAP_USD                      → write STOP, delete VMs, alert, exit 3
  hand-made eval < step-0 − 0.10 at any eval step ≥ 50 → write STOP, delete VMs, alert, exit 4
  no climb VM running                  → relaunch (zone rotation); after MAX_FAIL consecutive lives with no new
                                         checkpoint → alert, exit 5
Spend = finished VM lives (GCP operation log, ~/Code/infra/gcp/usage.py) + running VM elapsed × rate.

  nohup python3 infra/climb_loop.py > ~/.claude/delegations/sqlforge/climb1-loop.out 2>&1 &
  python3 infra/climb_loop.py --dry-run --once      # print decisions, launch nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = "sqlforge-bf3e24"
BUCKET = "gs://sqlforge-bf3e24-smoke/climb1"
PREFIX = "sqlforge-climb1"
ZONES = ["us-central1-b", "us-central1-c", "us-central1-f", "us-east1-b", "us-east1-d", "us-east4-b", "us-east4-c",
         "us-east5-a", "us-east5-b", "us-east5-c", "us-south1-a", "us-south1-b", "us-west1-a", "us-west1-b",
         "us-west1-c", "europe-west4-a", "europe-west4-b", "europe-west4-c", "europe-west1-b", "europe-west1-c",
         "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c", "asia-south1-c"]
CAP_USD = 40.0
RATE = 1.80            # $/h, conservative spot g4-standard-48 (us-east1 1.72, us-central1 1.77)
MAX_FAIL = 4
REGRESS_DROP, REGRESS_FROM = 0.10, 50
POLL = 60
STATE_DIR = Path.home() / ".claude/delegations/sqlforge"
STATE = STATE_DIR / "climb1-loop.json"
PIDFILE = STATE_DIR / "climb1-loop.pid"
STARTUP = Path(__file__).resolve().parent / "climb1-startup.sh"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
sys.path.insert(0, str(Path.home() / "Code/infra/gcp"))
import usage  # noqa: E402


def now_ist() -> str:
    return dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")


def log(msg: str) -> None:
    print(f"{now_ist()} {msg}", flush=True)


def sh(*args, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, check=check)


def gs_cat(path: str) -> str | None:
    r = sh("gcloud", "storage", "cat", f"{BUCKET}/{path}")
    return r.stdout if r.returncode == 0 else None


def alert(msg: str) -> None:
    log(f"ALERT {msg}")
    sh("osascript", "-e", f'display notification "{msg}" with title "climb1 loop" sound name "Glass"')
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        sh("curl", "-fsS", "-m", "10", "-d", f"climb1: {msg}", f"ntfy.sh/{topic}")


def climb_vms() -> list[dict]:
    r = sh("gcloud", "compute", "instances", "list", "--project", PROJECT, "--filter", f"name~^{PREFIX}",
           "--format", "json(name,zone.basename(),status,creationTimestamp)")
    return json.loads(r.stdout or "[]") if r.returncode == 0 else []


def spend() -> float:
    done = sum(r["est_usd"] for r in usage.runs(PROJECT) if r["instance"].startswith(PREFIX) and r["est_usd"] != "")
    running = 0.0
    for v in climb_vms():
        start = dt.datetime.fromisoformat(v["creationTimestamp"])
        running += (dt.datetime.now(dt.timezone.utc) - start).total_seconds() / 3600 * RATE
    return round(done + running, 2)


def delete_vms(dry: bool) -> None:
    for v in climb_vms():
        log(f"delete {v['name']} ({v['zone']})")
        if not dry:
            sh("gcloud", "compute", "instances", "delete", v["name"], "--zone", v["zone"], "--project", PROJECT, "--quiet")


def stop(reason: str, code: int, dry: bool):
    if not dry:
        sh("bash", "-c", f"echo '{reason} {now_ist()}' | gcloud storage cp - {BUCKET}/STOP")
    delete_vms(dry)
    alert(reason)
    sys.exit(code)


def regression() -> str | None:
    txt = gs_cat("runs/train-climb1-eval.jsonl")
    if not txt:
        return None
    rows = {}
    for line in txt.splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["eval_step"]] = r  # last row per step wins (resumes can repeat a step)
    if 0 not in rows or "handmade" not in rows[0].get("exec_acc_by_set", {}):
        return None
    base = rows[0]["exec_acc_by_set"]["handmade"]
    for s in sorted(rows):
        hm = rows[s].get("exec_acc_by_set", {}).get("handmade")
        if s >= REGRESS_FROM and hm is not None and hm < base - REGRESS_DROP:
            return f"hand-made eval regressed: step {s} {hm:.3f} vs step-0 {base:.3f}"
    return None


def launch(n: int, dry: bool) -> str | None:
    name = f"{PREFIX}-{n}"
    for z in ZONES:
        cmd = ["gcloud", "compute", "instances", "create", name, "--project", PROJECT, "--zone", z,
               "--machine-type", "g4-standard-48", "--provisioning-model", "SPOT",
               "--instance-termination-action", "DELETE", "--max-run-duration", "22h",
               "--image-family", "common-cu129-ubuntu-2204-nvidia-580", "--image-project", "deeplearning-platform-release",
               "--boot-disk-type", "hyperdisk-balanced", "--boot-disk-size", "200GB", "--scopes", "cloud-platform",
               "--labels", "sqlforge=true", "--metadata-from-file", f"startup-script={STARTUP}"]
        if dry:
            log(f"[dry-run] would create {name} in {z}")
            return z
        r = sh(*cmd)
        if r.returncode == 0:
            log(f"launched {name} in {z}")
            return z
        log(f"  {z}: {(r.stderr.strip().splitlines() or ['?'])[-1][:160]}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        PIDFILE.write_text(str(os.getpid()))
    st = json.loads(STATE.read_text()) if STATE.exists() else {"launches": 0, "fails": 0, "latest_at_launch": None,
                                                                  "no_capacity_rounds": 0}
    log(f"loop start (dry={a.dry_run}) cap=${CAP_USD} state={st}")
    while True:
        if gs_cat("DONE") is not None:
            alert("climb 1 finished (DONE marker)")
            sys.exit(0)
        marker = gs_cat("STOP")
        if marker is not None:
            delete_vms(a.dry_run)
            alert(f"stopped by STOP marker: {marker.strip()}")
            sys.exit(1)
        usd = spend()
        if usd >= CAP_USD:
            stop(f"spend cap reached: ${usd} >= ${CAP_USD}", 3, a.dry_run)
        reg = regression()
        if reg:
            stop(reg, 4, a.dry_run)
        vms = [v for v in climb_vms() if v["status"] in ("RUNNING", "PROVISIONING", "STAGING")]
        latest = (gs_cat("ckpt/LATEST") or "none").strip()
        if not vms:
            if st["launches"] > 0:
                st["fails"] = st["fails"] + 1 if latest == st["latest_at_launch"] else 0
            if st["fails"] >= MAX_FAIL:
                alert(f"{st['fails']} VM lives in a row made no checkpoint progress (LATEST={latest}); loop exiting")
                sys.exit(5)
            z = launch(st["launches"] + 1, a.dry_run)
            if z:
                st.update(launches=st["launches"] + 1, latest_at_launch=latest, no_capacity_rounds=0)
            else:
                st["no_capacity_rounds"] += 1
                log(f"no zone had spot capacity (round {st['no_capacity_rounds']})")
                if st["no_capacity_rounds"] == 6:
                    alert("no spot capacity in any zone for ~30 min; still retrying")
        log(f"status vms={[v['name'] + '@' + v['zone'] for v in vms]} LATEST={latest} spend=${usd} "
            f"launches={st['launches']} fails={st['fails']}")
        if not a.dry_run:
            STATE.write_text(json.dumps(st))
        if a.once:
            return
        time.sleep(POLL if vms else 300 if st["no_capacity_rounds"] else POLL)


if __name__ == "__main__":
    main()
