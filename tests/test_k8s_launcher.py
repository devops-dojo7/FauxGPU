import pytest

import api.k8s_launcher as k8s_launcher


def test_is_available_false_outside_cluster():
    # Tests always run outside a cluster (no incluster config file present).
    assert k8s_launcher.is_available() is False


def test_inference_available_false_outside_cluster():
    assert k8s_launcher.inference_available() is False


def test_launch_job_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="not available"):
        k8s_launcher.launch_job(
            model_label="custom",
            model_params={"params": 6.74e9, "num_layers": 32, "hidden_dim": 4096, "num_heads": 32, "head_dim": 128},
            gpu_id="h100-sxm",
            topology_shape="nvlink_node",
            gpus_per_node=8,
            num_nodes=1,
            fabric_id="infiniband-hdr",
            precision="bf16",
            tokens_per_step=32768,
            total_steps=50,
            speedup=200,
        )


def test_launch_inference_job_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="not available"):
        k8s_launcher.launch_inference_job(model_preset="llama2-7b", gpu_id="h100-sxm", precision="bf16")


def test_launch_inference_job_builds_manifests_when_in_cluster(monkeypatch):
    monkeypatch.setattr(k8s_launcher, "_in_cluster", True)
    monkeypatch.setenv("INFERENCE_SERVER_IMAGE", "simgpu/inference-server:dev")

    created = {}

    class FakeAppsV1Api:
        def create_namespaced_deployment(self, namespace, body):
            created["deployment"] = (namespace, body)

    class FakeCoreV1Api:
        def create_namespaced_service(self, namespace, body):
            created["service"] = (namespace, body)

    monkeypatch.setattr(k8s_launcher.client, "AppsV1Api", FakeAppsV1Api)
    monkeypatch.setattr(k8s_launcher.client, "CoreV1Api", FakeCoreV1Api)

    name = k8s_launcher.launch_inference_job(model_preset="llama2-7b", gpu_id="h100-sxm", precision="bf16")

    assert name.startswith("simgpu-inference-")
    dep_namespace, dep_body = created["deployment"]
    assert dep_namespace == "default"
    container = dep_body["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "simgpu/inference-server:dev"
    env_names = {e["name"] for e in container["env"]}
    assert {"MODEL_PRESET", "SIMGPU_MODEL", "PRECISION"} <= env_names
    svc_namespace, svc_body = created["service"]
    assert svc_namespace == "default"
    assert svc_body["metadata"]["name"] == name


def test_delete_inference_job_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="not available"):
        k8s_launcher.delete_inference_job("simgpu-inference-abc123")


def test_delete_inference_job_calls_k8s_apis(monkeypatch):
    monkeypatch.setattr(k8s_launcher, "_in_cluster", True)
    monkeypatch.setenv("INFERENCE_SERVER_IMAGE", "simgpu/inference-server:dev")

    deleted = {}

    class FakeAppsV1Api:
        def delete_namespaced_deployment(self, name, namespace):
            deleted["deployment"] = (name, namespace)

    class FakeCoreV1Api:
        def delete_namespaced_service(self, name, namespace):
            deleted["service"] = (name, namespace)

    monkeypatch.setattr(k8s_launcher.client, "AppsV1Api", FakeAppsV1Api)
    monkeypatch.setattr(k8s_launcher.client, "CoreV1Api", FakeCoreV1Api)

    k8s_launcher.delete_inference_job("simgpu-inference-abc123")

    assert deleted["deployment"] == ("simgpu-inference-abc123", "default")
    assert deleted["service"] == ("simgpu-inference-abc123", "default")


def test_launch_inference_job_passes_langfuse_env_when_set(monkeypatch):
    monkeypatch.setattr(k8s_launcher, "_in_cluster", True)
    monkeypatch.setenv("INFERENCE_SERVER_IMAGE", "simgpu/inference-server:dev")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    created = {}

    class FakeAppsV1Api:
        def create_namespaced_deployment(self, namespace, body):
            created["deployment"] = body

    class FakeCoreV1Api:
        def create_namespaced_service(self, namespace, body):
            pass

    monkeypatch.setattr(k8s_launcher.client, "AppsV1Api", FakeAppsV1Api)
    monkeypatch.setattr(k8s_launcher.client, "CoreV1Api", FakeCoreV1Api)

    k8s_launcher.launch_inference_job(model_preset="llama2-7b", gpu_id="h100-sxm", precision="bf16")

    env_names = {e["name"] for e in created["deployment"]["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert "LANGFUSE_PUBLIC_KEY" in env_names
    assert "LANGFUSE_SECRET_KEY" in env_names
