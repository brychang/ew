#!/bin/bash
#SBATCH --job-name=ribbon-seg-id
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Optional: uncomment and set your Princeton account if required.
# SBATCH --account=<YOUR_PI_OR_PROJECT>

# 1) Move to project directory.
cd /home/bc0043/ew

# 1.5) Preflight auth check for Graphene-backed CloudVolume access.
# This script uses a graphene:// path and requires this secret file.
SECRET_FILE="$HOME/.cloudvolume/secrets/cave-secret.json"
if [[ ! -f "$SECRET_FILE" ]]; then
  echo "ERROR: Missing Graphene credential: $SECRET_FILE" >&2
  echo "Create/copy cave-secret.json into ~/.cloudvolume/secrets/ before submitting." >&2
  exit 2
fi

# 1.6) Preflight DNS from compute node for the Graphene endpoint.
python - <<'PY'
import socket
import sys

host = "minnie.microns-daf.com"
try:
  ip = socket.gethostbyname(host)
except OSError as exc:
  print(
    "ERROR: Compute node cannot resolve minnie.microns-daf.com. "
    "This job needs external network/DNS access to the Graphene endpoint.",
    file=sys.stderr,
  )
  print(f"DNS failure detail: {exc}", file=sys.stderr)
  print(
    "Ask Princeton RC for an internet-enabled partition or proxy config, "
    "or run from a node with outbound DNS/network access.",
    file=sys.stderr,
  )
  raise SystemExit(3)
else:
  print(f"DNS preflight OK: {host} -> {ip}")
PY

# 2) Ensure dependencies are available in this environment.
# If already synced, this is fast/no-op.
uv sync

# 3) Run the parallel lookup script.
# Tune --workers to match --cpus-per-task.
uv run python temp-ribbon_seg_id.py \
  --input data/ribbon_v2_info.df \
  --output-dir data/ribbon_seg_id_runs \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunk-size 2000 \
  --format csv
