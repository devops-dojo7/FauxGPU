# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-25

Ten new simulation features, closing out the full feature roadmap, plus two
small UX/observability fixes.

### Added

- **What-if GPU recommender** — given a model and an objective (minimize
  cost or minimize time), searches GPU type × count × tensor/pipeline-parallel
  degree for feasible, ranked configurations.
- **Cost/utilization dashboard** — cumulative cost and GPU utilization over a
  training run's simulated timeline, including idle-GPU cost, derived from
  the run's own recorded steps.
- **Multi-tenant scheduler simulation** — a discrete-event, priority-based
  scheduler with preemption over a fixed GPU pool, visualized as a
  scheduling timeline (Gantt chart).
- **Failure/chaos injection** — inject GPU Xid errors, NVLink degradation,
  or node drain into a live training run and watch throughput/GPU count
  respond in real time, with an annotated event strip.
- **Checkpointing & fault-tolerant recovery** — periodic checkpoint-save
  overhead and, on a crash-inducing chaos event, the real cost of recovery
  (lost progress since the last checkpoint + restore time).
- **Broader accelerator coverage** — Google TPU v5e, v5p, and v6e
  (Trillium) added to the GPU catalog, alongside the existing AMD
  MI300X/MI325X and AWS Trainium3 entries, using Google's own published
  per-chip specs.
- **MIG-aware capacity planning** — a bin-packing visualizer for NVIDIA
  Multi-Instance GPU: pack tenant requests for MIG profiles across a pool
  of physical GPUs and see what fits, what doesn't, and the resulting
  compute/memory utilization.
- **Autoscaling simulation (HPA/KEDA-style)** — simulates a Kubernetes
  HorizontalPodAutoscaler control loop scaling inference replicas against a
  k6-shaped traffic pattern, including scale-down stabilization and
  backlog/queueing impact when under-provisioned.
- **Network/InfiniBand congestion modeling** — models bandwidth contention
  when multiple concurrent training jobs share one physical interconnect
  fabric, comparing isolated vs. contended communication time per job.
- **Job-queue trace replay** — replay a CSV job trace (arrival times, GPU
  counts, durations, priorities) through the same multi-tenant scheduler,
  with a bundled illustrative sample trace.

### Fixed

- VRAM breakdown panel now clarifies that its per-GPU split assumes ideal
  sharding independent of the configured tensor/pipeline-parallel degree,
  and points a capacity overflow at the Datacenter tab to scale the cluster
  up.
- The FauxGPU Grafana dashboard now surfaces the simulated GPU's
  human-readable model name (previously only the catalog id was available),
  plus a new "Active GPU / Runs" table panel.

## [0.1.0] - undated (accumulated prior to 0.2.0)

Initial release. Assembled over a number of earlier sessions before this
project's changelog started being tracked, so no single release date
applies — this entry summarizes everything the app already did going into
0.2.0.

### Added

- Core simulation engine (`engine/`): VRAM/KV-cache breakdown, training
  step-time and cost estimation, cluster topology (NVLink/InfiniBand),
  inference serving (prefill/decode, disaggregated serving, PagedAttention,
  speculative decoding).
- FastAPI backend wrapping the engine, with SSE streaming for live
  inference requests and an in-process live training-run simulator
  (including a real-k8s-Job launch path).
- Next.js web UI: Training, Inference (llm-d), Datacenter, Compare GPUs,
  Compare Models, and Playground tabs, with a shareable-link config export.
- Kubernetes/K3s layer: a fake GPU device plugin, Helm chart, GPU Operator
  Playground (`run-ai/fake-gpu-operator`), and Datacenter Mode with
  KWOK-simulated large node pools.
- Observability stack: Prometheus, a custom "FauxGPU" Grafana dashboard,
  the vendor NVIDIA DCGM dashboard, k6 load testing (and its own Grafana
  dashboard), and Langfuse tracing for inference requests.
- GPU catalog expanded to 34 GPUs across NVIDIA, AMD, Intel, AWS, Huawei,
  Tenstorrent, Meta, and Microsoft, and a 60-entry model preset list
  (dense, GQA, MoE, and MoE+MLA architectures).
- Rebranded to FauxGPU, with an editorial UI redesign and dark mode.
