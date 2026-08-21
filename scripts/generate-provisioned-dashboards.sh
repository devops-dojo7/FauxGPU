#!/usr/bin/env bash
# Grafana's file-based dashboard provisioner (used by docker-compose and the
# grafana-dashboards-configmap Helm template) does not resolve a dashboard's
# ${DS_...} __inputs template variables the way the "Import" UI flow does —
# it only substitutes those on manual import. nvidia-dcgm-dashboard.json is
# a vendored community export that still uses ${DS_PROMETHEUS}, so it must
# be pre-resolved to a fixed datasource uid before it can be provisioned
# from a file. Re-run this after updating either vendored dashboard JSON in
# k3s/grafana/.
set -euo pipefail
cd "$(dirname "$0")/.."

# Two copies of the same resolved output: one for the docker-compose
# Grafana's file provisioner, one inside the Helm chart directory (Helm's
# .Files.Get can only read files under the chart root, so
# grafana-dashboards-configmap.yaml needs its own copy).
OUT_DIRS=(
  k3s/observability/grafana-provisioning/dashboards/generated
  k3s/helm/simgpu/grafana-dashboards
)

for out_dir in "${OUT_DIRS[@]}"; do
  mkdir -p "$out_dir"
  for src in k3s/grafana/*.json; do
    name=$(basename "$src")
    sed 's/\${DS_PROMETHEUS}/grafanacloud-prom/g' "$src" > "$out_dir/$name"
    echo "generated $out_dir/$name"
  done
done
