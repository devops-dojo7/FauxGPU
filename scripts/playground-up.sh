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

# Heterogeneous fleet: one entry per k3d agent (this script always creates
# exactly 3), each simulating a different real GPU — so a GPU-aware tool
# (kubectl describe node, srelens-tui's :gpuinfo, etc.) sees a genuinely
# mixed datacenter rather than 3 identical nodes. Fields, colon-separated:
#   pool-name : gpuProduct : gpuCount : gpuMemory(MiB) : driverVersion : cudaVersion : gpuFamily : gpuMachine
# gpuProduct/gpuCount/gpuMemory go to fake-gpu-operator itself (it writes
# nvidia.com/gpu.product/count/memory from these); the rest go to this
# chart's own label-enricher Job (k3s/helm/simgpu's
# fakeGpuOperator.nodePools — see values.yaml), which fills in the
# driver/CUDA/family/machine labels fake-gpu-operator doesn't set itself.
POOLS=(
  "h100:H100-SXM5-80GB:8:81920:550.90.07:12.4:hopper:NVIDIA-DGX-H100"
  "a100:A100-SXM4-40GB:8:40960:535.129.03:12.2:ampere:NVIDIA-DGX-A100"
  "b200:B200-SXM-192GB:8:196608:570.86.10:12.8:blackwell:NVIDIA-DGX-B200"
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

echo "== assigning agent nodes to node pools (one GPU model per node) =="
# mapfile/readarray is bash 4+ only — macOS still ships bash 3.2, so build
# the array with a plain read loop instead.
agent_nodes=()
while IFS= read -r node; do
  agent_nodes+=("$node")
done < <(kubectl get nodes -o name | grep -E "k3d-${CLUSTER_NAME}-agent-" | sed 's#node/##' | sort)

fleet_values="$(mktemp)"
trap 'rm -f "$fleet_values"' EXIT
{
  echo "fakeGpuOperator:"
  echo "  nodePools:"
} > "$fleet_values"

fgo_pool_args=()
for i in "${!POOLS[@]}"; do
  IFS=: read -r name product count memory driver cuda family machine <<<"${POOLS[$i]}"
  node="${agent_nodes[$i]:-}"
  if [ -z "$node" ]; then
    echo "warning: no agent node left for pool '$name' (only ${#agent_nodes[@]} agent(s) in this cluster) — skipping it" >&2
    continue
  fi
  echo "  $node -> $name ($product)"
  kubectl label node "$node" run.ai/simulated-gpu-node-pool="$name" --overwrite >/dev/null

  fgo_pool_args+=(--set "topology.nodePools.${name}.gpuProduct=${product}")
  fgo_pool_args+=(--set "topology.nodePools.${name}.gpuCount=${count}")
  fgo_pool_args+=(--set "topology.nodePools.${name}.gpuMemory=${memory}")

  cat >>"$fleet_values" <<YAML
    - name: "${name}"
      driverVersion: "${driver}"
      cudaVersion: "${cuda}"
      gpuFamily: "${family}"
      gpuMachine: "${machine}"
YAML
done

echo "== installing fake-gpu-operator (simulates the real NVIDIA GPU Operator) =="
helm upgrade -i fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator \
  --namespace fake-gpu-operator --create-namespace --version "$FAKE_GPU_OPERATOR_VERSION" \
  --set runtimeClass.enabled=false \
  "${fgo_pool_args[@]}"

echo "== deploying the simgpu chart against fake-gpu-operator =="
kubectl delete job simgpu-trainer --ignore-not-found
helm upgrade -i simgpu k3s/helm/simgpu --set gpuBackend=fake-gpu-operator -f "$fleet_values"

echo "== waiting for simgpu-api and simgpu-web to be ready =="
kubectl rollout status deployment/simgpu-api --timeout=180s
kubectl rollout status deployment/simgpu-web --timeout=180s

cat <<EOF

Playground is up — a heterogeneous 3-node fleet (H100 / A100 / B200, one
model per node; see the POOLS list at the top of this script to change it).

Explore the simulated GPU Operator (each node reports its own model):
  kubectl get pods -n fake-gpu-operator
  kubectl get nodes -L run.ai/simulated-gpu-node-pool,nvidia.com/gpu.product
  kubectl describe node <agent-node> | grep -A12 nvidia.com/gpu

A GPU-aware Kubernetes TUI (e.g. srelens-tui's :gpuinfo) pointed at this
cluster's context ($(kubectl config current-context 2>/dev/null || echo "k3d-$CLUSTER_NAME")) will show
this same fleet as if it were real hardware.

Watch the sample trainer job:
  kubectl logs -f job/simgpu-trainer

Reach the API / website:
  kubectl port-forward svc/simgpu-api 8000:8000
  kubectl port-forward svc/simgpu-web 3000:3000

For the full guided walkthrough (including advanced GPU Operator topics),
open the site's Playground tab.

Tear down with: scripts/playground-down.sh
EOF
