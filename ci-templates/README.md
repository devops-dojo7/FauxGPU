# Reusable mock-datacenter CI template

`.github/workflows/simgpu-e2e-reusable.yml` in this repo is a
[reusable GitHub Actions workflow](https://docs.github.com/en/actions/using-workflows/reusing-workflows)
— it spins up a real k3d cluster, deploys this project's Helm chart onto it
(with either the built-in lightweight fake-GPU device-plugin or the
[fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator) integration),
runs `scripts/ci-smoke.sh` (GPU scheduling, a training run, a real inference
request round trip) plus the pytest suite, then tears the cluster down.

This repo dogfoods it directly — `.github/workflows/ci.yml`'s `e2e` job calls
it with `uses: ./.github/workflows/simgpu-e2e-reusable.yml`. Other repos
reference it the same way GitHub Actions supports referencing any reusable
workflow across repos: no copying required, just point `uses:` at this
repo's workflow file and pin a ref.

## Using it from your own repo

Add a thin caller workflow, e.g. `.github/workflows/mock-dc-test.yml`:

```yaml
name: Mock datacenter test

on:
  pull_request:
  workflow_dispatch:

jobs:
  e2e:
    uses: <org>/gpu-cluster-simulator/.github/workflows/simgpu-e2e-reusable.yml@main
    with:
      gpu-backend: simgpu # or fake-gpu-operator
```

Replace `<org>/gpu-cluster-simulator` with this repo's actual path once it
has a public remote, and pin `@main` to a tag/SHA for anything beyond
experimentation — reusable workflows follow the same ref-pinning tradeoffs
as any other Action.

## What it actually proves

`scripts/ci-smoke.sh` is the same script this repo's own CI runs — read it
directly rather than trusting a description here. In short: `GET /health`
and `/gpus` respond, `POST /runs/simulate` produces a run that reaches
`running`/`done`, `POST /inference/stream` reaches an `event: done`, and
`GET /metrics` exposes `simgpu_run_status`. Run it yourself against any live
deployment (`kubectl port-forward svc/simgpu-api 8000:8000` first):

```bash
API_URL=http://localhost:8000 ./scripts/ci-smoke.sh
```

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `gpu-backend` | `simgpu` | `simgpu` (built-in device-plugin) or `fake-gpu-operator` |
| `k3d-version` | `v5.7.4` | k3d release to install in the runner |
