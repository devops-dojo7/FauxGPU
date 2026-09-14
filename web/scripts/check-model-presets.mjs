#!/usr/bin/env node
// Structural sanity checks on MODEL_PRESETS (src/lib/types.ts) — catches
// data-entry mistakes (duplicate ids, an active_params that exceeds total
// params, a GQA head count that doesn't divide the head count, etc.)
// automatically, the same way tests/test_gpu_catalog_invariants.py does for
// the GPU side. Deliberately does not check individual numeric values
// against real model cards — that's a sourcing question for a human.
//
// Run with: node --experimental-strip-types scripts/check-model-presets.mjs
// (the flag is required on Node 22; Node 23.6+ strips inline TS by default,
// so it's a harmless no-op there too).

import { MODEL_PRESETS } from "../src/lib/types.ts";

const failures = [];
const fail = (msg) => failures.push(msg);

const ids = MODEL_PRESETS.map((p) => p.id);
const dupeIds = ids.filter((id, i) => ids.indexOf(id) !== i);
if (dupeIds.length > 0) fail(`Duplicate preset id(s): ${[...new Set(dupeIds)].join(", ")}`);

const labels = MODEL_PRESETS.map((p) => p.label);
const dupeLabels = labels.filter((l, i) => labels.indexOf(l) !== i);
if (dupeLabels.length > 0) fail(`Duplicate preset label(s): ${[...new Set(dupeLabels)].join(", ")}`);

for (const p of MODEL_PRESETS) {
  if (!/^[a-z0-9.-]+$/.test(p.id)) fail(`${p.id}: id must be lowercase [a-z0-9.-] only`);
  if (!p.label.trim()) fail(`${p.id}: empty label`);
  if (!(p.params > 0)) fail(`${p.id}: params must be > 0`);
  if (!(p.num_layers > 0)) fail(`${p.id}: num_layers must be > 0`);
  if (!(p.hidden_dim > 0)) fail(`${p.id}: hidden_dim must be > 0`);
  if (!(p.num_heads > 0)) fail(`${p.id}: num_heads must be > 0`);
  if (!(p.head_dim > 0)) fail(`${p.id}: head_dim must be > 0`);

  if (p.active_params != null) {
    if (!(p.active_params > 0)) fail(`${p.id}: active_params must be > 0 when set (MoE)`);
    if (p.active_params > p.params) fail(`${p.id}: active_params (${p.active_params}) exceeds total params (${p.params})`);
  }
  if (p.num_kv_heads != null) {
    if (!(p.num_kv_heads > 0)) fail(`${p.id}: num_kv_heads must be > 0 when set`);
    if (p.num_kv_heads > p.num_heads) fail(`${p.id}: num_kv_heads (${p.num_kv_heads}) exceeds num_heads (${p.num_heads})`);
  }
  if (p.kv_latent_dim != null && !(p.kv_latent_dim > 0)) fail(`${p.id}: kv_latent_dim must be > 0 when set`);
}

if (failures.length > 0) {
  console.error(`check-model-presets: ${failures.length} problem(s) found in MODEL_PRESETS:\n`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}

console.log(`check-model-presets: ${MODEL_PRESETS.length} presets OK`);
