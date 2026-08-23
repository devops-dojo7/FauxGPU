#!/usr/bin/env bash
# Tears down the GPU Operator Playground cluster created by
# scripts/playground-up.sh. Deleting the k3d cluster removes everything in
# it (fake-gpu-operator, the simgpu chart) — no separate helm uninstall
# needed.
set -euo pipefail

CLUSTER_NAME="simgpu-playground"

if ! command -v k3d >/dev/null 2>&1; then
  echo "k3d is not installed — nothing to tear down." >&2
  exit 1
fi

if k3d cluster list "$CLUSTER_NAME" >/dev/null 2>&1; then
  k3d cluster delete "$CLUSTER_NAME"
  echo "Deleted cluster '$CLUSTER_NAME'."
else
  echo "No cluster named '$CLUSTER_NAME' found — nothing to do."
fi
