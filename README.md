# GPU Cluster Simulator

Learn how GPU training and inference actually work — VRAM, KV cache,
NVLink/InfiniBand topology, [llm-d](https://llm-d.ai/)-style disaggregated
serving, and real cost tradeoffs — without needing real GPU hardware.

## Status

All phases are built and verified:
- **Phase 1:** core calculation engine + FastAPI service.
- **Phase 2:** Next.js website (GPU picker, VRAM/KV-cache calculator, topology visualizer).
- **Phase 3:** K3s fake-GPU-node layer (device plugin + Helm chart + sample trainer Job).
- **Phase 4:** inference/serving simulator modeled on [llm-d](https://llm-d.ai/) —
  prefill (compute-bound TTFT) vs. decode (memory-bandwidth-bound TPOT),
  prefix-cache-hit routing, and colocated-vs-disaggregated throughput
  comparison, plus a tabbed UI (Training / Inference) with a live training
  panel fed by the K3s trainer Job.
- **Phase 5:** interactive extras — well-lit-path presets, per-1K-token cost
  view, a live single-prompt playground (streaming TTFT/tokens-per-sec/KV
  cache, with an OOM check against the selected GPU's VRAM), a multi-request
  playground simulating continuous batching and prefill queueing across
  several concurrent prompts, shareable config links, and a button to launch
  a *real* Kubernetes Job (not just the in-process simulation) straight from
  the website.
- **Phase 6:** architecture-aware model formulas — GQA (`num_kv_heads`,
  shrinks KV cache below plain multi-head attention), MoE (`active_params`,
  separates the total params VRAM must hold from the smaller active count
  that drives compute/memory-bandwidth cost per token), and MLA
  (`kv_latent_dim`, DeepSeek-V3's compressed-KV-cache attention variant) —
  plus a vLLM PagedAttention toggle (block-rounded KV allocation) and a
  `gpu_memory_utilization`-bounded capacity estimate (max concurrent
  sequences a GPU can serve). New presets: Qwen2.5 7B/72B (GQA), NVIDIA
  Nemotron-4 340B, Qwen3 235B-A22B (MoE), DeepSeek-V3/R1 671B-A37B (MoE+MLA),
  Kimi K2 1T-A32B (MoE+MLA, Moonshot AI).
- **Phase 7:** tensor/pipeline parallelism + power draw + live GPU monitoring.
  `engine/parallelism.py` adds TP (splits per-layer matmuls, cutting compute
  ~linearly but adding a per-layer NVLink activation all-reduce) and PP
  (splits layers across GPUs, with a "pipeline bubble" idle-time penalty that
  shrinks as microbatches increase) on top of the existing DP gradient
  all-reduce — the three ways NCCL collectives show up in real distributed
  training (see
  [Lambda's intro to multi-GPU/multi-node NCCL training](https://lambda.ai/blog/introduction-multi-gpu-multi-node-distributed-training-nccl-2-0)
  for the DP/ring-all-reduce half of this). `engine/power.py` estimates GPU
  power draw (idle→TDP interpolation, phase-weighted for training's
  compute/comm split, a fixed HBM-traffic-driven estimate for decode) and
  feeds cost-panel energy/kWh totals. The Training and Inference tabs both
  got a "GPU monitoring (simulated)" section — small Grafana-style live
  sparkline charts (`components/Sparkline.tsx`) for power draw, SM
  utilization, NCCL network-busy %, and HBM memory-bus-busy %, fed by the
  same live training-run/prompt-playground state already being polled —
  not a real Grafana/Prometheus pipeline, a self-contained in-app view so
  simulated numbers never mix with any real infrastructure dashboards.
- **Phase 8:** speculative decoding, throughput/parallelism sweep curves, and
  a real Prometheus bridge. `engine/speculative.py` models a small draft
  model proposing `gamma` tokens per round, verified by the target model in
  one batched pass; expected tokens/round follows the standard geometric
  series in `acceptance_rate`, and — realistically — a weak draft model or
  high `gamma` can make this *slower* than plain decoding, not just faster.
  New draft-sized presets: Qwen2.5 0.5B, Llama-3.2 1B. `components/LineChart.tsx`
  is a small generic multi-series XY chart powering two new sweep panels:
  "Throughput vs. request rate" (Inference tab — colocated vs. disaggregated
  tokens/sec across a requests/sec sweep, showing the curve bend directly
  instead of one snapshot) and "Step time vs. tensor parallel degree"
  (Training tab — the compute-shrinks/TP-comm-grows crossover). `GET
  /metrics` on the API (`api/metrics.py`) exposes tracked runs in standard
  Prometheus exposition format (status/step/tokens/power gauges) — the
  bridge artifact for pointing a real Prometheus/Grafana at this simulator.
- **Phase 9:** real Grafana Cloud integration (`api/grafana_push.py`),
  entirely optional and off unless configured. Two separate credentials:
  a Grafana service-account token (`GRAFANA_URL`/`GRAFANA_API_KEY`) posts
  real start/finish **annotations** to Grafana via its HTTP API every time
  any run starts or finishes (in-process simulated runs, Helm-deployed
  sample trainer Job, and dynamically-launched real k8s Jobs all go through
  this); a Prometheus remote-write Access Policy token
  (`GRAFANA_REMOTE_WRITE_URL`/`GRAFANA_PROM_USERNAME`/`GRAFANA_PROM_TOKEN`,
  scoped to `metrics:write`) **pushes** live gauges (status/step/tokens/power)
  from a background task every 5s while any run is active — using the
  [`prometheus-remote-writer`](https://pypi.org/project/prometheus-remote-writer/)
  package. `GET /grafana/status` reports whether each is configured;
  `POST /grafana/push-now` triggers a manual push. Wired into the Helm
  chart as an optional Secret (`templates/grafana-secret.yaml`, only
  rendered when `grafana.enabled=true`) — real credentials are passed via
  `--set-string` at deploy time, never committed to `values.yaml`.
  Verified live end-to-end against the user's real Grafana Cloud instance:
  annotations for both an in-process simulated run and the real
  Helm-deployed trainer Job landed and were queried back successfully, and
  a manual push sent 4 real samples into the Prometheus/Mimir datasource.
- **Phase 10 (in progress):** mock-datacenter platform expansion — an opt-in
  `gpuBackend: fake-gpu-operator` mode (see "Datacenter mode" below) targeting
  [run-ai/fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator) for
  MIG/DRA and a 100+ node KWOK-simulated fleet, plus (planned) a full
  Grafana+Prometheus+Langfuse observability stack, real per-request inference
  tracing, and a reusable CI/CD template for testing against the mock
  cluster. See `/home/kumail/.claude/plans/transient-swimming-heron.md` for
  the full phased plan.

See `/home/kumail/.claude/plans/calm-fluttering-teapot.md` for the plan
covering phases 1-9.

## Layout

- `engine/` — pure Python calc package (no web/k8s deps). VRAM breakdown,
  KV cache, step-time, cost, cluster topology, and inference/serving
  (`inference.py` — prefill/decode/disaggregation, llm-d-style, plus vLLM
  PagedAttention + capacity planning) formulas. `ModelShape` (`memory.py`)
  supports dense/MHA models plus three optional architecture variants:
  GQA (`num_kv_heads`), MoE (`active_params`), and MLA (`kv_latent_dim`) —
  each defaults to plain-dense behavior when unset. GPU/vendor specs and
  pricing live in `engine/data/gpus.yaml`.
- `api/` — FastAPI service wrapping `engine/` as HTTP endpoints.
- `web/` — Next.js + TypeScript + Tailwind site, tabbed into two modes:
  - **Training** — model/GPU pickers, VRAM breakdown bar, topology diagram
    (React Flow), cost calculator, and a "Live training run" panel that
    polls the API for step-by-step progress reported by the K3s trainer Job.
  - **Inference (llm-d)** — an explainer of prefill vs. decode and
    disaggregated serving with a link to llm-d.ai, well-lit-path presets,
    controls (prompt/output length, decode batch size, requests/sec,
    prefix-cache hit %, decode pool GPU count), a colocated-vs-disaggregated
    throughput comparison, cost/1K-tokens, a live single-prompt playground
    (`lib/simEngine.ts` mirrors the Python prefill/decode formulas client-side
    so it can stream token-by-token with no per-token network round trip),
    and a multi-request playground simulating continuous batching.
  - The "Live training run" panel can also launch a fast in-process
    simulated run, or (when the API has k8s RBAC — see below) a **real**
    Kubernetes Job via `POST /runs/launch-k8s-job`.
- `tests/` — formula sanity checks against known reference numbers (e.g.
  Llama-2 7B VRAM/KV-cache footprint).
- `k3s/device-plugin/` — Go fake Kubernetes device plugin. Registers a
  configurable number of simulated `simgpu.dev/gpu` devices with the kubelet
  so pods can request them like real GPUs and be scheduled/observed via
  `kubectl describe node`. No real hardware involved — NVLink/InfiniBand
  differences stay as modeled numbers, not real traffic shaping.
- `k3s/trainer/` — sample "distributed trainer" Job. Uses the same `engine/`
  formulas to work out step time for the requested model/GPU/topology, then
  sleeps a scaled-down version of that time and logs structured JSON
  progress — watch a simulated multi-GPU training run actually get
  scheduled onto the fake GPU resources. If `API_URL` is set (the Helm
  chart wires it to the in-cluster API automatically), it also POSTs
  progress to `/runs/{run_id}/...` so the website's "Live training run"
  panel can show it — best-effort, so a stopped/unreachable API never
  fails the run.
- `k3s/helm/simgpu/` — Helm chart deploying the device plugin (DaemonSet),
  API, web, trainer Job, and RBAC (`templates/rbac.yaml` — a ServiceAccount +
  Role/RoleBinding letting the API pod create batch/v1 Jobs, used by the
  website's "Launch real k8s Job" button via `api/k8s_launcher.py`).

## Running the website + API locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
.venv/bin/uvicorn api.main:app --reload

cd web
npm install
npm run dev
```

API docs: http://localhost:8000/docs · Website: http://localhost:3000

## Running the K3s layer

Verified against a local [k3d](https://k3d.io) cluster:

```bash
# build the images
docker build -t simgpu/device-plugin:dev -f k3s/device-plugin/Dockerfile k3s/device-plugin
docker build -t simgpu/api:dev -f api/Dockerfile .
docker build -t simgpu/trainer:dev -f k3s/trainer/Dockerfile .
docker build -t simgpu/inference-server:dev -f k3s/inference-server/Dockerfile .
docker build -t simgpu/web:dev web

# spin up a cluster and load the images
k3d cluster create simgpu --agents 2 --wait
k3d image import simgpu/device-plugin:dev simgpu/api:dev simgpu/trainer:dev simgpu/inference-server:dev simgpu/web:dev -c simgpu

# deploy
helm install simgpu k3s/helm/simgpu

# see the simulated GPU resources show up on every node
kubectl describe nodes | grep -A5 simgpu.dev/gpu

# watch the sample trainer job get scheduled and simulate training steps
kubectl logs -f job/simgpu-trainer

# reach the API / website
kubectl port-forward svc/simgpu-api 8000:8000
kubectl port-forward svc/simgpu-web 3000:3000
```

Change `values.yaml` (`trainer.topologyShape`, `trainer.gpuModel`,
`trainer.fabricId`, `devicePlugin.gpuCount`, etc.) and `helm upgrade` to see
step time and communication overhead change with the simulated topology.

Tear down with `k3d cluster delete simgpu`.

## Datacenter mode: fake-gpu-operator

The default `gpuBackend: simgpu` above is a minimal fake device-plugin —
enough to demonstrate real k8s GPU-resource scheduling, but only a single
static GPU model/count per node. For a richer "how does a real GPU
datacenter operate" experience — MIG partitioning, Dynamic Resource
Allocation (DRA, k8s 1.31+), and its own Prometheus GPU-utilization
metrics — this project can instead target
[run-ai/fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator)
(Apache 2.0), installed as a prerequisite:

```bash
# install fake-gpu-operator itself (a separate Helm release, its own namespace)
helm install fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator \
  --namespace fake-gpu-operator --create-namespace \
  --set topology.nodePools.default.gpuProduct=H100-SXM5-80GB \
  --set topology.nodePools.default.gpuCount=8 \
  --set topology.nodePools.default.gpuMemory=81920

# label the nodes it should treat as GPU nodes (matches the pool above)
kubectl label node k3d-simgpu-agent-0 run.ai/simulated-gpu-node-pool=default
kubectl label node k3d-simgpu-agent-1 run.ai/simulated-gpu-node-pool=default

# point this chart at it instead of the built-in device-plugin
helm upgrade simgpu k3s/helm/simgpu --set gpuBackend=fake-gpu-operator
```

When `gpuBackend: fake-gpu-operator`, this chart stops deploying its own
device-plugin DaemonSet and instead requests
`fakeGpuOperator.resourceName` (default `nvidia.com/gpu`, matching
fake-gpu-operator's mock-NVML DaemonSet) on the API and trainer pods —
`fakeGpuOperator.nodePoolLabelKey`/`nodePool` must match whatever pool you
configured in fake-gpu-operator's own `topology.nodePools` values above.

### Optional: 100+ node simulated fleet (KWOK)

fake-gpu-operator can register hundreds of fake `Node` objects backed by
[KWOK](https://kwok.sigs.k8s.io/) (no real kubelet) — enough to make
`kubectl get nodes` and fleet-wide Grafana panels feel like a real
hyperscaler-sized cluster. This is for experiencing scheduling/topology at
scale, not for running real trainer/inference Jobs across all of them —
KWOK nodes can't run real pods, so this project's own workloads still only
ever land on the small real node pool from the section above.

```bash
# install the KWOK controller (once per cluster)
KWOK_VERSION=v0.7.0
kubectl apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/kwok.yaml"
kubectl apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/stage-fast.yaml"

# enable fake-gpu-operator's KWOK device plugin
helm upgrade fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator \
  --namespace fake-gpu-operator --reuse-values \
  --set kwokGpuDevicePlugin.enabled=true

# render 100 fake GPU nodes
helm upgrade simgpu k3s/helm/simgpu \
  --set gpuBackend=fake-gpu-operator \
  --set fleet.enabled=true \
  --set fleet.numNodes=100

kubectl get nodes -l type=kwok
```

## Observability stack: Grafana + Prometheus + Langfuse

A self-contained observability stack you run yourself, no external Grafana
Cloud account needed (that's the separate section below).

### docker-compose (local dev, no cluster needed)

```bash
docker compose up                                             # simulator only: api :8000, web :3000
docker compose --profile observability up                     # + prometheus :9090, grafana :3001
docker compose --profile langfuse up                          # + langfuse :3002 and its storage stack
docker compose --profile observability --profile langfuse up  # everything
```

Grafana comes pre-provisioned with the same two dashboards used by the K3s
Grafana Cloud integration (`k3s/grafana/dashboard.json`,
`nvidia-dcgm-dashboard.json`) via the resolved copies in
`k3s/observability/grafana-provisioning/dashboards/generated/` — regenerate
those (and the Helm chart's copy) with
`./scripts/generate-provisioned-dashboards.sh` after editing either
vendored dashboard JSON. Langfuse is vendored from its own published
compose file at `k3s/observability/langfuse/docker-compose.langfuse.yml`
(see that file's header for the two port remaps applied to avoid colliding
with this project's own :3000/:9090). On first Langfuse boot, open
http://localhost:3002, create an org/project, and copy the public/secret
key pair into `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` env vars for the
`api` service to enable tracing (see "Langfuse tracing" below once wired
up).

### K8s / Helm (in-cluster)

Requires two upstream charts installed first — this project only adds a
`ServiceMonitor` and dashboard `ConfigMap` on top rather than reimplementing
either:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

helm repo add langfuse https://langfuse.github.io/langfuse-k8s
helm install langfuse langfuse/langfuse --namespace langfuse --create-namespace

helm upgrade simgpu k3s/helm/simgpu --set observability.enabled=true
```

`observability.enabled=true` renders a `ServiceMonitor` for `simgpu-api`
(picked up automatically by kube-prometheus-stack's Prometheus Operator)
and a `ConfigMap` wrapping the same two dashboards, labeled
`grafana_dashboard: "1"` for kube-prometheus-stack's Grafana sidecar to
auto-load.

### Langfuse tracing of individual inference requests

Everything above is infra-level gauges (power, step, tokens/sec). Langfuse
traces individual LLM *requests* — TTFT/decode spans, tokens/sec, cost per
prompt — teaching request-level LLM observability on top of the simulated
numbers. Off by default; wire it up the same way as Grafana Cloud below:

```bash
helm upgrade simgpu k3s/helm/simgpu \
  --set langfuse.enabled=true \
  --set-string langfuse.publicKey=pk-lf-... \
  --set-string langfuse.secretKey=sk-lf-... \
  --set-string langfuse.host=http://langfuse-web.langfuse.svc:3000  # or your Langfuse Cloud host
```

Locally (docker-compose or plain `uvicorn`), just set
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` env vars for the
`api` process. Check `GET /langfuse/status` to confirm it's wired up. Two
things get traced, both via `api/langfuse_client.py`:

- **The website's playgrounds** (Live prompt playground, Multi-request
  playground) — `POST /inference/stream` (SSE) is what makes these send
  real prompt/timing data to the backend at all; each request that
  completes emits one Langfuse trace with a `prefill` span and a `decode`
  generation.
- **Real workloads against the mock cluster** — `k3s/inference-server`
  (built from `k3s/inference-server/Dockerfile`) is a real, long-running
  FastAPI process exposing an OpenAI-Completions-shaped `/v1/completions`
  endpoint backed by the same simulated timing, launched on demand via
  `POST /inference/launch-k8s-server` (mirrors the existing
  `POST /runs/launch-k8s-job` training-Job launcher, but as a
  Deployment+Service since it's long-running, not one-shot). Point a real
  client — `curl`, the `openai` SDK — at the resulting Service and its
  requests produce real Langfuse traces flowing through the simulated
  GPU-backed infra.

## Optional: Grafana Cloud integration

Off by default. To enable, get two credentials from your Grafana Cloud org
(**Administration → Service accounts** for the first, the **Prometheus**
connection page under your stack on grafana.com for the second) and deploy
with:

```bash
helm upgrade simgpu k3s/helm/simgpu \
  --set grafana.enabled=true \
  --set-string grafana.url=https://yourorg.grafana.net \
  --set-string grafana.apiKey=glsa_... \
  --set-string grafana.remoteWriteUrl=https://prometheus-prod-XX-.../api/prom/push \
  --set-string grafana.promUsername=123456 \
  --set-string grafana.promToken=glc_...
```

Every run start/finish posts a real annotation; while any run is active, a
background task pushes live power/step/token gauges via Prometheus
remote-write every 5s. Check `GET /grafana/status` to confirm both are
wired up. Never put real values in `values.yaml` — pass them via `--set-string`
only, same as any other secret.

A ready-made dashboard model lives at `k3s/grafana/dashboard.json` — Power
Draw, Tokens Seen, Run Status, Step Progress, Currently Running, plus a
`simgpu`-tagged annotation overlay, all against the `grafanacloud-prom`
datasource uid (rename that uid in the file if your org's default
Prometheus datasource differs). Create it with:

```bash
curl -X POST "$GRAFANA_URL/api/dashboards/db" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @k3s/grafana/dashboard.json
```

### Vendor-standard DCGM dashboard

`api/dcgm_metrics.py` also emits real `DCGM_FI_*` metric names/labels
(`Hostname`/`gpu`/`UUID`/`modelName`) — the same shape real
`nvidia-dcgm-exporter` deployments publish — approximated from the
simulator's own compute/power model (temperature from power draw, VRAM
used/free from a fixed utilization assumption, GR/tensor/DRAM "active"
fractions from the compute-vs-communication time split, energy from
power × elapsed time; XID errors and PCIe replays always 0 — no fault
injection modeled). Only emitted for currently-`running` runs, one series
set per simulated GPU.

This means the standard community
[NVIDIA DCGM Dashboard for Kubernetes](https://grafana.com/grafana/dashboards/23382-nvidia-mig-dcgm/)
renders simulator data directly. The model (with `${DS_PROMETHEUS}` input
placeholder) is saved at `k3s/grafana/nvidia-dcgm-dashboard.json`; import it
bound to your Prometheus datasource with:

```bash
python3 -c "
import json
d = json.load(open('k3s/grafana/nvidia-dcgm-dashboard.json'))
d['id'] = None
payload = {'dashboard': d, 'overwrite': False, 'folderId': 0,
           'inputs': [{'name': 'DS_PROMETHEUS', 'type': 'datasource',
                       'pluginId': 'prometheus', 'value': 'grafanacloud-prom'}]}
json.dump(payload, open('/tmp/dcgm-import.json', 'w'))
"
curl -X POST "$GRAFANA_URL/api/dashboards/import" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/dcgm-import.json
```

**GPU selection and live load simulation:** the GPU picked in the website
(top of either tab) is exactly what's used when you click "Start new
training run" or "Launch real k8s Job" — that `gpu_id` becomes `meta.gpu`
on the run, which drives every DCGM/`simgpu_run_*` series (model name, TDP,
memory bandwidth, everything). While a run is `running`, `api/live_phase.py`
tracks which phase of the *current* step the run is actually in (compute
vs. communication) from its own observed real-time pace, and every push
reports the instantaneous power/temperature/engine-activity for whichever
phase it's in — not a flat time-weighted average. The result is a real
pulsing load on the dashboard, synced to the run's actual speedup: verified
by sampling `DCGM_FI_DEV_POWER_USAGE` at 2s resolution during a live run and
seeing it genuinely alternate between ~300W (compute phase) and ~240W
(communication phase), not a flat line.
