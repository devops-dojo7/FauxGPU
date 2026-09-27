# FauxGPU

**Simulate a real GPU datacenter with zero GPUs.**

FauxGPU deploys a fake-GPU Kubernetes layer — real `nvidia.com/gpu`- or
`simgpu.dev/gpu`-style extended resources, a real scheduled trainer Job, a
real long-running inference server, and a real FastAPI + Next.js UI — so
you can practice Kubernetes GPU scheduling, watch a training run progress,
and hit a real inference API, all without any actual GPU hardware. Every
number the simulator reports (VRAM breakdown, step time, KV-cache growth,
cost) is computed from the same formulas that govern real hardware, not
hardcoded.

Full project docs, architecture, and a live demo:
[devops-dojo7.github.io/FauxGPU](https://devops-dojo7.github.io/FauxGPU/) ·
Source: [github.com/devops-dojo7/FauxGPU](https://github.com/devops-dojo7/FauxGPU)

## Installing this chart

```console
helm install simgpu oci://ghcr.io/devops-dojo7/fauxgpu/charts/fauxgpu --version 0.2.4
```

That's it — every component image (device-plugin, api, trainer,
inference-server, web) is pulled straight from this project's own published
GHCR images, no `docker build` step required.

Check the simulated GPU resources show up on every node, then watch the
bundled sample trainer Job schedule and run:

```console
kubectl describe nodes | grep -A5 simgpu.dev/gpu
kubectl logs -f job/simgpu-trainer
kubectl port-forward svc/simgpu-api 8000:8000
kubectl port-forward svc/simgpu-web 3000:3000
```

## Configuration

**`values.yaml` is the single file that configures the whole deployment** —
every knob for every component lives there, grouped by concern and
commented in place:

- `image.registry` / `image.tag` — shared defaults every component's own
  `image.repository`/`tag`/`pullPolicy` falls back to. Bump one version in
  one place to move the whole stack, or override a single component to
  point it somewhere else (a custom-built image, a different registry).
- `gpuBackend` — `simgpu` (the lightweight built-in fake device-plugin,
  default) or `fake-gpu-operator` (points at an already-installed
  [run-ai/fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator)
  for MIG partitioning, DRA, and its own Prometheus GPU metrics).
- `devicePlugin.*` / `fakeGpuOperator.*` — simulated GPU model, count, and
  (for the fake-gpu-operator backend) heterogeneous multi-pool fleets with
  full NVIDIA GPU Operator/NFD label parity.
- `fleet.*` — opt-in 100+ node KWOK-simulated fleet for scheduling/topology
  at datacenter scale (requires `gpuBackend: fake-gpu-operator` and KWOK
  already installed).
- `trainer.*` — the bundled sample training Job's model preset, GPU
  topology (NVLink/InfiniBand shape, GPUs per node, node count), precision,
  and step count.
- `api.*` / `web.*` / `inferenceServer.*` — replica counts, ports, resource
  requests/limits, and enable/disable flags for the core services.
- `api.autoscaling` — opt-in `HorizontalPodAutoscaler` for `simgpu-api`
  (requires a metrics-server in-cluster, bundled by default in k3d/k3s).
- `networkPolicy.enabled` — opt-in default-deny `NetworkPolicy` for
  `simgpu-api`/`simgpu-web` with explicit allows for this chart's own
  traffic; `kubectl port-forward` is unaffected either way.
- `grafana.*` / `langfuse.*` / `observability.*` — optional integrations,
  every one `enabled: false` by default and a no-op until configured. Real
  credentials should always be passed via `--set-string` at deploy time,
  never committed to a values file:

  ```console
  helm upgrade simgpu oci://ghcr.io/devops-dojo7/fauxgpu/charts/fauxgpu \
    --set langfuse.enabled=true \
    --set-string langfuse.publicKey=pk-lf-... \
    --set-string langfuse.secretKey=sk-lf-...
  ```

Building the images from source instead of pulling the published ones?
Clone the repo and see `k3s/helm/simgpu/values-dev.yaml` and the README's
"Build the images yourself" section for the local-build override.

See the full [README](https://github.com/devops-dojo7/FauxGPU#readme) for
the guided GPU Operator Playground script, the 100+ node datacenter mode
walkthrough, and the observability stack setup.
