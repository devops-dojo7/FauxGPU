#!/usr/bin/env bash
# Brings up the GPU Operator Playground: a local k3d cluster (1 server + 3
# agents — BYOI, runs entirely on your own machine) with run-ai's
# fake-gpu-operator simulating the real NVIDIA GPU Operator's node-level
# GPU visibility, plus this repo's simgpu chart deployed on top. Mirrors
# the manual walkthrough in README.md's "Datacenter mode" section — that
# section stays as the reference; this script is the guided one-command
# path used by the website's Playground tab. Companion to
# scripts/playground-down.sh.
set -euo pipefail

CLUSTER_NAME="simgpu-playground"
FAKE_GPU_OPERATOR_VERSION="0.2.0"
IMAGES=(
  simgpu/device-plugin:dev
  simgpu/api:dev
  simgpu/trainer:dev
  simgpu/inference-server:dev
  simgpu/web:dev
)

echo "== checking required tools =="
missing=()
for tool in docker k3d kubectl helm; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required tool(s): ${missing[*]}" >&2
  echo "Install them, then re-run this script." >&2
  exit 1
fi

echo "== checking images are built =="
missing_images=()
for image in "${IMAGES[@]}"; do
  docker image inspect "$image" >/dev/null 2>&1 || missing_images+=("$image")
done
if [ "${#missing_images[@]}" -gt 0 ]; then
  echo "Missing image(s): ${missing_images[*]}" >&2
  cat >&2 <<'EOF'
Build them first:
  docker build -t simgpu/device-plugin:dev -f k3s/device-plugin/Dockerfile k3s/device-plugin
  docker build -t simgpu/api:dev -f api/Dockerfile .
  docker build -t simgpu/trainer:dev -f k3s/trainer/Dockerfile .
  docker build -t simgpu/inference-server:dev -f k3s/inference-server/Dockerfile .
  docker build -t simgpu/web:dev web
EOF
  exit 1
fi

echo "== creating k3d cluster: 1 server + 3 agents =="
if k3d cluster list "$CLUSTER_NAME" >/dev/null 2>&1; then
  echo "Cluster '$CLUSTER_NAME' already exists — reusing it."
else
  k3d cluster create "$CLUSTER_NAME" --servers 1 --agents 3 --wait
fi

echo "== importing images into the cluster =="
k3d image import "${IMAGES[@]}" -c "$CLUSTER_NAME"

echo "== labeling agent nodes for fake-gpu-operator =="
agent_nodes=$(kubectl get nodes -o name | grep -E "k3d-${CLUSTER_NAME}-agent-" | sed 's#node/##')
# shellcheck disable=SC2086
kubectl label node $agent_nodes run.ai/simulated-gpu-node-pool=default --overwrite

echo "== installing fake-gpu-operator (simulates the real NVIDIA GPU Operator) =="
helm upgrade -i fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator \
  --namespace fake-gpu-operator --create-namespace --version "$FAKE_GPU_OPERATOR_VERSION" \
  --set topology.nodePools.default.gpuProduct=H100-SXM5-80GB \
  --set topology.nodePools.default.gpuCount=8 \
  --set topology.nodePools.default.gpuMemory=81920 \
  --set runtimeClass.enabled=false

echo "== deploying the simgpu chart against fake-gpu-operator =="
kubectl delete job simgpu-trainer --ignore-not-found
helm upgrade -i simgpu k3s/helm/simgpu --set gpuBackend=fake-gpu-operator

echo "== waiting for simgpu-api and simgpu-web to be ready =="
kubectl rollout status deployment/simgpu-api --timeout=180s
kubectl rollout status deployment/simgpu-web --timeout=180s

cat <<EOF

Playground is up.

Explore the simulated GPU Operator:
  kubectl get pods -n fake-gpu-operator
  kubectl describe node <agent-node> | grep -A5 nvidia.com/gpu

Watch the sample trainer job:
  kubectl logs -f job/simgpu-trainer

Reach the API / website:
  kubectl port-forward svc/simgpu-api 8000:8000
  kubectl port-forward svc/simgpu-web 3000:3000

For the full guided walkthrough (including advanced GPU Operator topics),
open the site's Playground tab.

Tear down with: scripts/playground-down.sh
EOF
