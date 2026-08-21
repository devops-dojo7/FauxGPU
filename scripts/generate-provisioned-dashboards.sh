#!/usr/bin/env bash
# The vendored dashboards in k3s/grafana/ are in two different shapes and
# neither is directly consumable by Grafana's file-based provisioner (used
# by docker-compose and the grafana-dashboards-configmap Helm template):
#
#   - dashboard.json is wrapped in the *HTTP API* import shape
#     ({"dashboard": {...}, "overwrite": true} — what README's
#     `POST /api/dashboards/db` curl command for Grafana Cloud expects) but
#     the file provisioner wants the raw dashboard object directly, no
#     wrapper — otherwise it fails with "Dashboard title cannot be empty".
#   - nvidia-dcgm-dashboard.json is already raw, but still references its
#     datasource via a ${DS_PROMETHEUS} __inputs template variable, which
#     only Grafana's "Import" UI flow resolves — file provisioning leaves
#     it as a literal unresolved string.
#
# This unwraps the API-import shape when present and resolves
# ${DS_PROMETHEUS} to a fixed datasource uid either way. Re-run after
# updating either vendored dashboard JSON in k3s/grafana/.
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
    python3 -c "
import json, sys
with open('$src') as f:
    d = json.load(f)
d = d['dashboard'] if 'dashboard' in d and 'title' not in d else d
text = json.dumps(d, indent=2)
text = text.replace('\${DS_PROMETHEUS}', 'grafanacloud-prom')
sys.stdout.write(text)
" > "$out_dir/$name"
    echo "generated $out_dir/$name"
  done
done
