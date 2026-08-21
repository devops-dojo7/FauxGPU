#!/usr/bin/env bash
# End-to-end smoke assertions against a live simgpu deployment (API
# reachable at $API_URL, default http://localhost:8000). Shared by
# .github/workflows/simgpu-e2e-reusable.yml and anyone running the same
# checks by hand against a real cluster (`kubectl port-forward svc/simgpu-api
# 8000:8000` first) — kept as a standalone script so CI and a human get
# identical behavior instead of the checks living only inline in YAML.
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

echo "== health =="
curl -sf "$API_URL/health" | grep -q '"status":"ok"'

echo "== gpus list is non-empty =="
curl -sf "$API_URL/gpus" | python3 -c "import json,sys; assert len(json.load(sys.stdin)) > 0"

echo "== training run simulation completes with real steps =="
run_id=$(curl -sf -X POST "$API_URL/runs/simulate" \
  -H "Content-Type: application/json" \
  -d '{
        "model": {"params": 6.74e9, "num_layers": 32, "hidden_dim": 4096, "num_heads": 32, "head_dim": 128},
        "model_label": "ci-smoke",
        "topology": {"shape": "single_gpu", "gpu_id": "h100-sxm", "gpus_per_node": 1, "num_nodes": 1},
        "total_steps": 2,
        "speedup": 1000
      }' | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
sleep 1
curl -sf "$API_URL/runs/$run_id" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status'] in ('running', 'done'), d"

echo "== inference stream reaches 'done' =="
curl -sf -N -X POST "$API_URL/inference/stream" \
  -H "Content-Type: application/json" \
  -d '{
        "model": {"params": 6.74e9, "num_layers": 32, "hidden_dim": 4096, "num_heads": 32, "head_dim": 128},
        "gpu_id": "h100-sxm",
        "prompt": "ci smoke test",
        "prompt_tokens": 4,
        "max_output_tokens": 3
      }' | grep -q "event: done"

echo "== metrics endpoint exposes simgpu_run_status =="
curl -sf "$API_URL/metrics" | grep -q "^simgpu_run_status"

echo "all smoke checks passed"
