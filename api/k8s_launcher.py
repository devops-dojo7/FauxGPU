"""Launches real Kubernetes workloads from inside the API pod: batch/v1 Jobs
for the K3s trainer, and apps/v1 Deployments + Service for the long-running
k3s/inference-server.

This only works when the API is actually running in a cluster with the RBAC
granted by k3s/helm/simgpu/templates/rbac.yaml (ServiceAccount + Role letting
it create/delete Jobs, Deployments, and Services) — running
`uvicorn api.main:app` locally, this is simply unavailable, and callers
should check `is_available()`/`inference_available()` first.
"""

from __future__ import annotations

import os
import uuid

try:
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    _KUBERNETES_IMPORT_OK = True
except ImportError:  # pragma: no cover - kubernetes package always in requirements.txt
    _KUBERNETES_IMPORT_OK = False

_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

_in_cluster = False
if _KUBERNETES_IMPORT_OK:
    try:
        config.load_incluster_config()
        _in_cluster = True
    except ConfigException:
        _in_cluster = False


def _namespace() -> str:
    try:
        with open(_NAMESPACE_FILE) as f:
            return f.read().strip()
    except OSError:
        return "default"


def is_available() -> bool:
    return _in_cluster and bool(os.environ.get("TRAINER_IMAGE"))


def inference_available() -> bool:
    return _in_cluster and bool(os.environ.get("INFERENCE_SERVER_IMAGE"))


def launch_job(
    model_label: str,
    model_params: dict,
    gpu_id: str,
    topology_shape: str,
    gpus_per_node: int,
    num_nodes: int,
    fabric_id: str | None,
    precision: str,
    tokens_per_step: int,
    total_steps: int,
    speedup: float,
) -> str:
    if not is_available():
        raise RuntimeError("k8s Job launching is not available (not running in-cluster, or TRAINER_IMAGE unset)")

    job_name = f"simgpu-trainer-{uuid.uuid4().hex[:8]}"
    resource_name = os.environ.get("SIMGPU_RESOURCE_NAME", "simgpu.dev/gpu")
    trainer_image = os.environ["TRAINER_IMAGE"]

    env_list = [
        {"name": "MODEL_PRESET", "value": "custom"},
        {"name": "MODEL_LABEL", "value": model_label},
        {"name": "PARAMS", "value": str(model_params["params"])},
        {"name": "NUM_LAYERS", "value": str(model_params["num_layers"])},
        {"name": "HIDDEN_DIM", "value": str(model_params["hidden_dim"])},
        {"name": "NUM_HEADS", "value": str(model_params["num_heads"])},
        {"name": "HEAD_DIM", "value": str(model_params["head_dim"])},
        *([{"name": "NUM_KV_HEADS", "value": str(model_params["num_kv_heads"])}] if model_params.get("num_kv_heads") else []),
        *([{"name": "ACTIVE_PARAMS", "value": str(model_params["active_params"])}] if model_params.get("active_params") else []),
        *([{"name": "KV_LATENT_DIM", "value": str(model_params["kv_latent_dim"])}] if model_params.get("kv_latent_dim") else []),
        {"name": "SIMGPU_MODEL", "value": gpu_id},
        {"name": "TOPOLOGY_SHAPE", "value": topology_shape},
        {"name": "SIMGPU_COUNT", "value": str(gpus_per_node)},
        {"name": "NUM_NODES", "value": str(num_nodes)},
        {"name": "FABRIC_ID", "value": fabric_id or "infiniband-hdr"},
        {"name": "PRECISION", "value": precision},
        {"name": "TOKENS_PER_STEP", "value": str(tokens_per_step)},
        {"name": "TOTAL_STEPS", "value": str(total_steps)},
        {"name": "SPEEDUP", "value": str(speedup)},
        {"name": "API_URL", "value": f"http://simgpu-api:{os.environ.get('API_PORT', '8000')}"},
        {"name": "RUN_ID", "value": job_name},
    ]

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "labels": {"app": "simgpu-trainer", "launched-by": "simgpu-api"}},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": {"app": "simgpu-trainer"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "trainer",
                            "image": trainer_image,
                            "env": env_list,
                            "resources": {"limits": {resource_name: str(gpus_per_node)}},
                        }
                    ],
                },
            },
        },
    }

    client.BatchV1Api().create_namespaced_job(namespace=_namespace(), body=manifest)
    return job_name


def delete_job(job_name: str) -> None:
    """Stops a running trainer Job. Background propagation also deletes its
    Pod, matching what `kubectl delete job` does by default."""
    if not is_available():
        raise RuntimeError("k8s Job deletion is not available (not running in-cluster, or TRAINER_IMAGE unset)")
    client.BatchV1Api().delete_namespaced_job(name=job_name, namespace=_namespace(), propagation_policy="Background")


def launch_inference_job(model_preset: str, gpu_id: str, precision: str) -> str:
    """Unlike launch_job()'s one-shot batch/v1 Job, the inference server is
    long-running — a Deployment + Service, matching k3s/inference-server's
    real HTTP server (see api/routers/inference.py for the callers)."""
    if not inference_available():
        raise RuntimeError("inference server launching is not available (not running in-cluster, or INFERENCE_SERVER_IMAGE unset)")

    name = f"simgpu-inference-{uuid.uuid4().hex[:8]}"
    inference_image = os.environ["INFERENCE_SERVER_IMAGE"]

    env_list = [
        {"name": "MODEL_PRESET", "value": model_preset},
        {"name": "SIMGPU_MODEL", "value": gpu_id},
        {"name": "PRECISION", "value": precision},
    ]
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        env_list.append({"name": "LANGFUSE_PUBLIC_KEY", "value": os.environ["LANGFUSE_PUBLIC_KEY"]})
        env_list.append({"name": "LANGFUSE_SECRET_KEY", "value": os.environ["LANGFUSE_SECRET_KEY"]})
        env_list.append({"name": "LANGFUSE_HOST", "value": os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")})

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": {"app": name, "launched-by": "simgpu-api"}},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {"containers": [{"name": "inference-server", "image": inference_image, "env": env_list, "ports": [{"containerPort": 9000}]}]},
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name},
        "spec": {"selector": {"app": name}, "ports": [{"port": 9000, "targetPort": 9000}]},
    }

    namespace = _namespace()
    client.AppsV1Api().create_namespaced_deployment(namespace=namespace, body=deployment)
    client.CoreV1Api().create_namespaced_service(namespace=namespace, body=service)
    return name


def delete_inference_job(name: str) -> None:
    """Stops a running inference server Deployment + its Service."""
    if not inference_available():
        raise RuntimeError("inference server deletion is not available (not running in-cluster, or INFERENCE_SERVER_IMAGE unset)")
    namespace = _namespace()
    client.AppsV1Api().delete_namespaced_deployment(name=name, namespace=namespace)
    client.CoreV1Api().delete_namespaced_service(name=name, namespace=namespace)
