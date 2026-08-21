# Contributing to ErsatzGPU

Thanks for considering a contribution. This is a learning tool for GPU
training/inference math and Kubernetes GPU scheduling — contributions that
keep it simple, well-tested, and honest about what's simulated vs. real are
especially welcome.

## Project layout

- `engine/` — pure-Python calculation package (VRAM, KV cache, topology,
  cost, inference/serving formulas). No web or Kubernetes dependencies.
- `api/` — FastAPI service exposing `engine/` over HTTP, plus optional
  integrations (Grafana Cloud, Langfuse) that no-op when unconfigured.
- `web/` — Next.js frontend.
- `k3s/` — the K3s/Helm fake-GPU simulation layer (device-plugin, trainer,
  inference-server, Helm chart, Grafana dashboards).
- `tests/` — pytest suite for `engine/` and `api/`.
- `.github/workflows/` — CI (fast unit tests + web build on every push/PR)
  and a reusable end-to-end workflow that deploys the whole stack to a real
  k3d cluster (see `ci-templates/README.md`).

See `README.md` for the full local dev setup.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
.venv/bin/uvicorn api.main:app --reload

cd web
npm install
npm run dev
```

Before opening a PR, make sure these all pass:

```bash
.venv/bin/pytest tests/ -v          # Python
cd web && npx tsc --noEmit -p . && npm run lint && npm run build   # TypeScript
```

CI runs the same checks (`.github/workflows/ci.yml`) plus, on a schedule or
when a PR is labeled `e2e`, the full k3d cluster deploy
(`.github/workflows/simgpu-e2e-reusable.yml`). If your change touches the
Helm chart or `api/k8s_launcher.py`, add the `e2e` label to your PR so that
workflow actually exercises it before merge.

## Adding a new engine formula or API endpoint

- `engine/` functions must stay pure — no I/O, no web-framework imports.
  Add a test in `tests/` alongside the existing formula tests
  (`test_inference.py`, `test_topology_compute_cost.py`, etc.) rather than
  only exercising it through the API.
- New Pydantic request/response models go in `api/schemas.py`; new routes
  are grouped by concern under `api/routers/`.
- Optional external integrations (anything that talks to a real external
  service) should follow the shape of `api/grafana_push.py` /
  `api/langfuse_client.py`: read config from env vars, expose an
  `*_available()` check, and make every call best-effort — a misconfigured
  or unreachable integration must never break the simulator itself.

## Changing the Helm chart / K3s layer

- Every new capability should follow the existing `enabled:`-flag-per-block
  pattern in `k3s/helm/simgpu/values.yaml` (see `grafana`, `observability`,
  `langfuse`, `fleet` for examples) so it's opt-in and doesn't change
  default behavior.
- Validate with `helm template k3s/helm/simgpu` across the relevant flag
  combinations before opening a PR — CI's `e2e`-labeled path deploys to a
  real cluster, but a quick local render catches most mistakes faster.
- Real secrets never belong in `values.yaml` — pass them via `--set-string`
  at deploy time, matching the `grafana.*`/`langfuse.*` convention.

## Commit style

Commit messages should explain *why*, not just *what* — the diff already
shows what changed. A short summary line, then a body covering the
motivation and (for anything non-trivial) how it was verified, is the
pattern used throughout this repo's history.

## Reporting bugs / proposing features

Open an issue describing the problem or idea before a large PR — for
anything that changes the shape of `engine/` formulas, the Helm chart's
values contract, or adds a new external integration, a short discussion
up front saves rework. Small, focused PRs (one formula, one endpoint, one
chart flag) are much easier to review than large ones.

## Code of conduct

Be respectful and constructive. Assume good faith, disagree on substance
rather than people, and keep discussion focused on the project.
