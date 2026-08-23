"use client";

import { ReactNode, useState } from "react";
import { Card } from "./ui";

/** A terminal-styled, copyable command block. Presentational only — this
 * mirrors scripts/playground-*.sh and the README's k3d walkthrough, it
 * never executes anything itself. Wiring a browser button to run commands
 * on the host would need an unauthenticated local agent process; this app
 * has no auth anywhere, so that's a footgun this component deliberately
 * avoids. */
function TerminalBlock({ title, lines }: { title: string; lines: string[] }) {
  const [copied, setCopied] = useState(false);
  const text = lines.filter((l) => !l.startsWith("#")).join("\n");

  const copy = () => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  };

  return (
    <div className="rounded-lg overflow-hidden border border-hairline-strong">
      <div className="flex items-center justify-between px-3 py-1.5 bg-surface-strong">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
          </div>
          <span className="text-xs text-muted font-mono">{title}</span>
        </div>
        <button
          onClick={copy}
          className="text-xs font-medium text-muted hover:text-ink transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="text-xs font-mono p-3 overflow-x-auto bg-[#0b0e14] text-emerald-300 leading-relaxed">
        {lines.map((line, i) => (
          <div key={i} className={line.startsWith("#") ? "text-slate-500" : undefined}>
            {line === "" ? " " : line.startsWith("#") ? line : `$ ${line}`}
          </div>
        ))}
      </pre>
    </div>
  );
}

function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-hairline pt-4 mt-4 first:border-t-0 first:pt-0 first:mt-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-sm font-medium text-ink hover:text-muted transition-colors"
      >
        <span className={`transition-transform ${open ? "rotate-90" : ""}`}>{"›"}</span>
        {title}
      </button>
      {open && <div className="mt-4">{children}</div>}
    </div>
  );
}

function TopicCard({ name, blurb, href }: { name: string; blurb: string; href: string }) {
  return (
    <div className="rounded-lg border border-hairline p-3">
      <div className="text-sm font-medium text-ink mb-1">{name}</div>
      <p className="text-xs text-muted mb-2 leading-relaxed">{blurb}</p>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs font-medium text-body-strong hover:underline"
      >
        Official docs {"→"}
      </a>
    </div>
  );
}

const OPERATOR_CONFIG_TOPICS = [
  {
    name: "Multi-Instance GPU (MIG)",
    blurb: "Partitions one physical GPU into several fully-isolated instances, each with its own memory and compute slice.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html",
  },
  {
    name: "Time-Slicing GPUs",
    blurb: "Shares a whole GPU across multiple workloads sequentially, oversubscribing it without hardware partitioning.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html",
  },
  {
    name: "DRA driver for NVIDIA GPUs",
    blurb: "Kubernetes' newer Dynamic Resource Allocation API for expressing richer GPU requests than the plain device-plugin model.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-dra.html",
  },
  {
    name: "GPUDirect RDMA / Storage",
    blurb: "Lets NICs and storage read/write GPU memory directly, skipping a CPU-memory bounce for high-throughput multi-node training.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpudirect-rdma.html",
  },
  {
    name: "CDI and NRI support",
    blurb: "Container Device Interface / Node Resource Interface — newer, more portable ways for a runtime to expose devices to containers.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/cdi.html",
  },
];

const SANDBOXED_TOPICS = [
  {
    name: "Kata Containers",
    blurb:
      "Runs a pod inside a lightweight VM instead of a plain namespace, giving GPU workloads a hardware-enforced isolation boundary. Needs KVM and a real GPU — not something this fake-GPU playground can demonstrate meaningfully.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-kata.html",
  },
  {
    name: "Confidential Containers",
    blurb: "Extends Kata-style VM isolation with hardware-backed confidential computing, so even the host can't inspect workload memory.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/confidential-containers.html",
  },
  {
    name: "KubeVirt GPU passthrough / DRA",
    blurb: "Gives a full virtual machine (not just a container) direct or vGPU access to a real GPU, managed as a Kubernetes object.",
    href: "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/kubevirt.html",
  },
];

const PREREQS = [
  { name: "Docker", href: "https://docs.docker.com/get-docker/" },
  { name: "k3d", href: "https://k3d.io/#installation" },
  { name: "kubectl", href: "https://kubernetes.io/docs/tasks/tools/#kubectl" },
  { name: "Helm", href: "https://helm.sh/docs/intro/install/" },
];

export function PlaygroundPanel() {
  return (
    <div className="flex flex-col gap-6">
      <Card title="GPU Operator Playground">
        <p className="text-sm text-body leading-relaxed mb-3">
          Spin up a real local Kubernetes cluster — 1 control-plane + 3 workers — and deploy{" "}
          <a
            href="https://github.com/run-ai/fake-gpu-operator"
            target="_blank"
            rel="noopener noreferrer"
            className="text-body-strong hover:underline"
          >
            run-ai/fake-gpu-operator
          </a>{" "}
          on it, standing in for{" "}
          <a
            href="https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html"
            target="_blank"
            rel="noopener noreferrer"
            className="text-body-strong hover:underline"
          >
            NVIDIA&apos;s real GPU Operator
          </a>
          . This is <strong className="text-ink">BYOI — Bring Your Own Infrastructure</strong>: every command
          below runs on your own machine, nothing here is hosted for you.
        </p>
        <div className="flex flex-wrap gap-3">
          {PREREQS.map((p) => (
            <a
              key={p.name}
              href={p.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-medium px-3 py-1.5 rounded-full border border-hairline-strong text-body hover:bg-surface-strong transition-colors"
            >
              {p.name} {"↗"}
            </a>
          ))}
        </div>
      </Card>

      <Card title="Step 1 — Build the images">
        <p className="text-sm text-muted mb-3">Build the five images this repo ships.</p>
        <TerminalBlock
          title="terminal"
          lines={[
            "docker build -t simgpu/device-plugin:dev -f k3s/device-plugin/Dockerfile k3s/device-plugin",
            "docker build -t simgpu/api:dev -f api/Dockerfile .",
            "docker build -t simgpu/trainer:dev -f k3s/trainer/Dockerfile .",
            "docker build -t simgpu/inference-server:dev -f k3s/inference-server/Dockerfile .",
            "docker build -t simgpu/web:dev web",
          ]}
        />
      </Card>

      <Card title="Step 2 — Spin up the cluster">
        <p className="text-sm text-muted mb-3">
          One command creates the k3d cluster, installs fake-gpu-operator, and deploys the simgpu chart on top of
          it.
        </p>
        <TerminalBlock title="terminal" lines={["./scripts/playground-up.sh"]} />
        <Disclosure title="Show me what this script does">
          <TerminalBlock
            title="scripts/playground-up.sh"
            lines={[
              "# 1 control-plane + 3 workers",
              "k3d cluster create simgpu-playground --servers 1 --agents 3 --wait",
              "k3d image import simgpu/device-plugin:dev simgpu/api:dev simgpu/trainer:dev simgpu/inference-server:dev simgpu/web:dev -c simgpu-playground",
              "",
              "# status-updater only reacts to already-labeled nodes",
              "kubectl label node <agent-nodes> run.ai/simulated-gpu-node-pool=default",
              "",
              "# simulates the real NVIDIA GPU Operator's node-level GPU visibility",
              "helm upgrade -i fake-gpu-operator oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator \\",
              "  --namespace fake-gpu-operator --create-namespace --version 0.2.0 \\",
              "  --set topology.nodePools.default.gpuProduct=H100-SXM5-80GB \\",
              "  --set topology.nodePools.default.gpuCount=8",
              "",
              "helm upgrade -i simgpu k3s/helm/simgpu --set gpuBackend=fake-gpu-operator",
            ]}
          />
        </Disclosure>
      </Card>

      <Card title="Step 3 — Explore the GPU Operator">
        <p className="text-sm text-muted mb-3">
          fake-gpu-operator&apos;s status-updater and device-plugin components simulate the same node-level GPU
          resource that the real GPU Operator&apos;s driver-validator, device-plugin, and dcgm-exporter components
          publish — no real driver, no real GPU.
        </p>
        <TerminalBlock
          title="terminal"
          lines={[
            "kubectl get pods -n fake-gpu-operator",
            "kubectl describe node <agent-node> | grep -A5 nvidia.com/gpu",
          ]}
        />
      </Card>

      <Card title="Step 4 — Schedule a real workload">
        <p className="text-sm text-muted mb-3">
          Port-forward the API/website, then use the Training tab&apos;s &quot;Launch real k8s Job&quot; button — it
          now schedules against this cluster instead of the in-process simulation.
        </p>
        <TerminalBlock
          title="terminal"
          lines={["kubectl port-forward svc/simgpu-api 8000:8000", "kubectl port-forward svc/simgpu-web 3000:3000", "kubectl logs -f job/simgpu-trainer"]}
        />
      </Card>

      <Card title="Step 5 — Tear down">
        <TerminalBlock title="terminal" lines={["./scripts/playground-down.sh"]} />
      </Card>

      <Card title="Advanced: GPU Operator internals">
        <p className="text-sm text-muted mb-4">
          The real GPU Operator&apos;s driver-level config surface. This playground&apos;s fake-GPU backend
          simulates resource visibility and scheduling, not driver partitioning — these are explainers with links
          to the official docs, not scripted install steps here.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {OPERATOR_CONFIG_TOPICS.map((t) => (
            <TopicCard key={t.name} {...t} />
          ))}
        </div>
      </Card>

      <Card title="Advanced: Sandboxed workloads">
        <p className="text-sm text-muted mb-4">
          VM-level isolation for GPU workloads. These need KVM and a real GPU for the security boundary they
          provide, so they aren&apos;t part of the scripted playground above — read up here, then try them on real
          hardware.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {SANDBOXED_TOPICS.map((t) => (
            <TopicCard key={t.name} {...t} />
          ))}
        </div>
      </Card>
    </div>
  );
}
