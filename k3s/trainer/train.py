"""Sample "distributed trainer" for the simulated GPU cluster. Runs no real
compute — it uses the same engine/ formulas as the website to work out how
long each step *would* take on the requested GPU/topology, then sleeps that
long (scaled down by SPEEDUP so a demo run finishes in seconds instead of
real training time) and logs structured progress. The point is to watch a
multi-pod "training job" actually get scheduled onto the simulated GPU
resources the fake device plugin advertises, with step time changing when
you change topology/fabric — not to reproduce real numbers.

If API_URL is set, progress is also POSTed to the API's /runs endpoints so
the website can show it live. Reporting is best-effort: a stopped/unreachable
API never fails the run — the logs printed to stdout are always the source
of truth (`kubectl logs -f job/simgpu-trainer`).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/app")

from engine.compute import estimate_step_time  # noqa: E402
from engine.memory import ModelShape  # noqa: E402
from engine.topology import build_topology  # noqa: E402

MODEL_PRESETS = {
    "llama2-7b": ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128),
    "llama2-13b": ModelShape(params=13.0e9, num_layers=40, hidden_dim=5120, num_heads=40, head_dim=128),
    "llama2-70b": ModelShape(params=70.0e9, num_layers=80, hidden_dim=8192, num_heads=64, head_dim=128),
}


def env(key, default):
    return os.environ.get(key, default)


def report(api_url, run_id, path, payload):
    if not api_url:
        return
    url = f"{api_url.rstrip('/')}/runs/{run_id}{path}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=3).close()
    except (urllib.error.URLError, OSError) as e:
        print(json.dumps({"event": "report_failed", "path": path, "error": str(e)}), flush=True)


def main():
    model_id = env("MODEL_PRESET", "llama2-7b")
    if model_id == "custom":
        num_kv_heads = os.environ.get("NUM_KV_HEADS")
        active_params = os.environ.get("ACTIVE_PARAMS")
        kv_latent_dim = os.environ.get("KV_LATENT_DIM")
        model = ModelShape(
            params=float(env("PARAMS", "6.74e9")),
            num_layers=int(env("NUM_LAYERS", "32")),
            hidden_dim=int(env("HIDDEN_DIM", "4096")),
            num_heads=int(env("NUM_HEADS", "32")),
            head_dim=int(env("HEAD_DIM", "128")),
            num_kv_heads=int(num_kv_heads) if num_kv_heads else None,
            active_params=float(active_params) if active_params else None,
            kv_latent_dim=int(kv_latent_dim) if kv_latent_dim else None,
        )
    else:
        model = MODEL_PRESETS[model_id]
    model_label = env("MODEL_LABEL", model_id)

    gpu_id = env("SIMGPU_MODEL", "h100-sxm")
    shape = env("TOPOLOGY_SHAPE", "nvlink_node")
    gpus_per_node = int(env("SIMGPU_COUNT", "8"))
    num_nodes = int(env("NUM_NODES", "1"))
    fabric_id = env("FABRIC_ID", "infiniband-hdr")
    precision = env("PRECISION", "bf16")
    tokens_per_step = int(env("TOKENS_PER_STEP", "32768"))
    total_steps = int(env("TOTAL_STEPS", "50"))
    speedup = float(env("SPEEDUP", "200"))  # shrink simulated wall-clock so demos finish fast

    api_url = env("API_URL", "")
    run_id = env("RUN_ID", f"run-{int(time.time())}")

    topo = build_topology(shape, gpu_id, gpus_per_node=gpus_per_node, num_nodes=num_nodes, fabric_id=fabric_id)
    step = estimate_step_time(model, topo, tokens_per_step=tokens_per_step, precision=precision)

    start_payload = {
        "model": model_label,
        "gpu": gpu_id,
        "topology": shape,
        "total_gpus": topo.total_gpus,
        "compute_s_per_step": round(step.compute_s, 4),
        "communication_s_per_step": round(step.communication_s, 4),
        "total_s_per_step": round(step.total_s, 4),
        "total_steps": total_steps,
    }
    print(json.dumps({"event": "start", "run_id": run_id, **start_payload}), flush=True)
    report(api_url, run_id, "/start", start_payload)

    start = time.time()
    tokens_seen = 0
    for i in range(1, total_steps + 1):
        time.sleep(step.total_s / speedup)
        tokens_seen += tokens_per_step
        step_payload = {"step": i, "tokens_seen": tokens_seen, "elapsed_s": round(time.time() - start, 2)}
        print(json.dumps({"event": "step", "run_id": run_id, **step_payload}), flush=True)
        report(api_url, run_id, "/step", step_payload)

    print(json.dumps({"event": "done", "run_id": run_id, "total_steps": total_steps, "tokens_seen": tokens_seen}), flush=True)
    report(api_url, run_id, "/done", {})


if __name__ == "__main__":
    main()
