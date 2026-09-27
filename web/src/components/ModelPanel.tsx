"use client";

import { getModelPresetTier, MODEL_PRESET_GROUPS, MODEL_PRESETS, ModelShape } from "@/lib/types";
import { Card, ConfidenceBadge, Field, ModelArchBadges, NumberInput, Select, Toggle } from "./ui";

export interface ModelPanelState {
  presetId: string;
  model: ModelShape;
  precision: string;
  batchSize: number;
  seqLen: number;
  optimizer: string;
  fp32MasterCopy: boolean;
  checkpointing: boolean;
  training: boolean;
  customName: string;
  zeroStage: number;
  dpSize: number;
  peftMethod: string;
  peftRank: number;
  peftTargetModules: number;
}

/** Display name for a model config — the custom name if set, else the matching preset's label. */
export function getModelLabel(state: Pick<ModelPanelState, "presetId" | "customName">): string {
  if (state.presetId === "custom") return state.customName.trim() || "Custom";
  return MODEL_PRESETS.find((p) => p.id === state.presetId)?.label ?? state.presetId;
}

export function ModelPanel({
  state,
  onChange,
}: {
  state: ModelPanelState;
  onChange: (s: ModelPanelState) => void;
}) {
  const set = (patch: Partial<ModelPanelState>) => onChange({ ...state, ...patch });

  const onPreset = (id: string) => {
    const preset = MODEL_PRESETS.find((p) => p.id === id);
    if (id === "custom" || !preset) {
      set({ presetId: "custom" });
      return;
    }
    set({ presetId: id, model: { ...preset } });
  };

  return (
    <Card title="Model">
      <div className="grid grid-cols-2 gap-3">
        <div className="col-span-2">
          <Field label="Preset">
            <Select value={state.presetId} onChange={onPreset}>
              {MODEL_PRESET_GROUPS.map((g) => (
                <optgroup key={g.tier} label={g.label}>
                  {g.presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </optgroup>
              ))}
              <option value="custom">Custom</option>
            </Select>
          </Field>
        </div>

        {state.presetId === "custom" && (
          <div className="col-span-2">
            <Field label="Model name">
              <input
                type="text"
                className="rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-sm text-ink outline-none transition-colors focus:border-ink focus:border-2 focus:px-[11px] focus:py-[7px]"
                value={state.customName}
                onChange={(e) => set({ customName: e.target.value })}
                placeholder="e.g. My-Finetune-7B"
              />
            </Field>
          </div>
        )}

        <div className="col-span-2 flex flex-wrap gap-2 items-center">
          <ModelArchBadges model={state.model} />
          <ConfidenceBadge tier={getModelPresetTier(state.presetId)} />
        </div>

        <Field label="Params (billions)">
          <NumberInput
            value={state.model.params / 1e9}
            min={0.01}
            step={0.1}
            onChange={(v) => set({ presetId: "custom", model: { ...state.model, params: v * 1e9 } })}
          />
        </Field>
        <Field label="Layers">
          <NumberInput
            value={state.model.num_layers}
            min={1}
            onChange={(v) => set({ presetId: "custom", model: { ...state.model, num_layers: v } })}
          />
        </Field>
        <Field label="Hidden dim">
          <NumberInput
            value={state.model.hidden_dim}
            min={1}
            onChange={(v) => set({ presetId: "custom", model: { ...state.model, hidden_dim: v } })}
          />
        </Field>
        <Field label="Attention heads">
          <NumberInput
            value={state.model.num_heads}
            min={1}
            onChange={(v) => set({ presetId: "custom", model: { ...state.model, num_heads: v } })}
          />
        </Field>
        <Field label="Head dim">
          <NumberInput
            value={state.model.head_dim}
            min={1}
            onChange={(v) => set({ presetId: "custom", model: { ...state.model, head_dim: v } })}
          />
        </Field>
        <Field label="Precision">
          <Select value={state.precision} onChange={(v) => set({ precision: v })}>
            <option value="fp32">fp32</option>
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
            <option value="fp8">fp8</option>
          </Select>
        </Field>
        <Field label="Batch size">
          <NumberInput value={state.batchSize} min={1} onChange={(v) => set({ batchSize: v })} />
        </Field>
        <Field label="Sequence length">
          <NumberInput value={state.seqLen} min={1} step={128} onChange={(v) => set({ seqLen: v })} />
        </Field>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-hairline pt-4">
        <Toggle checked={state.training} onChange={(v) => set({ training: v })} label="Training (vs. inference-only)" />
        {state.training && (
          <>
            <Field label="Optimizer">
              <Select value={state.optimizer} onChange={(v) => set({ optimizer: v })}>
                <option value="adam">Adam</option>
                <option value="sgd_momentum">SGD + momentum</option>
                <option value="sgd">SGD</option>
              </Select>
            </Field>
            <Toggle checked={state.fp32MasterCopy} onChange={(v) => set({ fp32MasterCopy: v })} label="fp32 master weights" />
            <Toggle checked={state.checkpointing} onChange={(v) => set({ checkpointing: v })} label="Activation checkpointing" />
          </>
        )}
      </div>

      {state.training && (
        <div className="mt-3 flex flex-wrap items-end gap-4 border-t border-hairline pt-4">
          <Field label="ZeRO / FSDP sharding">
            <Select
              value={String(state.zeroStage)}
              onChange={(v) => set({ zeroStage: Number(v), peftMethod: Number(v) > 0 ? "full" : state.peftMethod })}
            >
              <option value="0">Off (full replicas)</option>
              <option value="1">Stage 1 — optimizer states (Pos)</option>
              <option value="2">Stage 2 — + gradients (Pos+g)</option>
              <option value="3">Stage 3 — + params (Pos+g+p, FSDP full)</option>
            </Select>
          </Field>
          {state.zeroStage > 0 && (
            <Field label="DP replicas sharded across">
              <NumberInput value={state.dpSize} min={1} max={4096} onChange={(v) => set({ dpSize: v })} />
            </Field>
          )}
        </div>
      )}

      {state.training && (
        <div className="mt-3 flex flex-wrap items-end gap-4 border-t border-hairline pt-4">
          <Field label="Fine-tuning method">
            <Select
              value={state.peftMethod}
              onChange={(v) => set({ peftMethod: v, zeroStage: v !== "full" ? 0 : state.zeroStage })}
            >
              <option value="full">Full fine-tuning (every param trainable)</option>
              <option value="lora">LoRA (Hu et al. 2021) — frozen base + tiny adapter</option>
              <option value="qlora">QLoRA (Dettmers et al. 2023) — 4-bit NF4 base + LoRA</option>
            </Select>
          </Field>
          {state.peftMethod !== "full" && (
            <>
              <Field label="LoRA rank (r)">
                <NumberInput value={state.peftRank} min={1} max={256} onChange={(v) => set({ peftRank: v })} />
              </Field>
              <Field label="Adapted matrices">
                <Select value={String(state.peftTargetModules)} onChange={(v) => set({ peftTargetModules: Number(v) })}>
                  <option value="1">1 (Wq only)</option>
                  <option value="2">2 (Wq, Wv — paper&apos;s default)</option>
                  <option value="3">3 (Wq, Wk, Wv)</option>
                  <option value="4">4 (Wq, Wk, Wv, Wo)</option>
                </Select>
              </Field>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
