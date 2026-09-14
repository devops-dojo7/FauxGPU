from fastapi import APIRouter, HTTPException

from api.schemas import (
    CostRequest,
    CostResponse,
    InferenceRequest,
    InferenceResponse,
    RecommendationCandidateOut,
    RecommendRequest,
    RecommendResponse,
    SpeculativeDecodingRequest,
    SpeculativeDecodingResponse,
    VramRequest,
    VramResponse,
)
from engine.cost import estimate_training_cost
from engine.gpu_specs import get_gpu
from engine.inference import estimate_serving_capacity, simulate_serving
from engine.memory import ModelShape, compute_vram_breakdown
from engine.parallelism import estimate_parallel_step_time
from engine.power import training_step_power_watts
from engine.recommend import recommend_configurations
from engine.speculative import simulate_speculative_decoding
from engine.topology import build_topology

router = APIRouter(prefix="/calculate", tags=["calculate"])


@router.post("/vram", response_model=VramResponse)
def calculate_vram(req: VramRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        breakdown = compute_vram_breakdown(
            model,
            precision=req.precision,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            optimizer=req.optimizer,
            fp32_master_copy=req.fp32_master_copy,
            checkpointing=req.checkpointing,
            training=req.training,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return VramResponse(
        weights_gb=breakdown.weights_gb,
        gradients_gb=breakdown.gradients_gb,
        optimizer_states_gb=breakdown.optimizer_states_gb,
        activations_gb=breakdown.activations_gb,
        kv_cache_gb=breakdown.kv_cache_gb,
        total_gb=breakdown.total_gb,
    )


@router.post("/cost", response_model=CostResponse)
def calculate_cost(req: CostRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        topo = build_topology(
            shape=req.topology.shape,
            gpu_id=req.topology.gpu_id,
            gpus_per_node=req.topology.gpus_per_node,
            num_nodes=req.topology.num_nodes,
            fabric_id=req.topology.fabric_id,
        )
        step = estimate_parallel_step_time(
            model,
            topo,
            tokens_per_step=req.tokens_per_step,
            precision=req.precision,
            utilization=req.utilization,
            tp_degree=req.tp_degree,
            pp_degree=req.pp_degree,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            num_microbatches=req.num_microbatches,
        )
        if step.total_s == float("inf"):
            raise ValueError(
                f"Tensor parallelism (TP={req.tp_degree}) needs NVLink, but {topo.gpu.name} has none."
            )
        cost = estimate_training_cost(
            topo, step, req.tokens_per_step, req.total_training_tokens, total_gpus_override=step.total_gpus
        )
        power_per_gpu = training_step_power_watts(
            topo.gpu, step.compute_s, step.dp_communication_s + step.tp_communication_s + step.pipeline_bubble_s, req.utilization
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    total_power_kw = power_per_gpu * step.total_gpus / 1000
    total_energy_kwh = total_power_kw * cost.total_time_hours

    return CostResponse(
        compute_s_per_step=step.compute_s,
        communication_s_per_step=step.dp_communication_s,
        tp_communication_s_per_step=step.tp_communication_s,
        pipeline_bubble_s_per_step=step.pipeline_bubble_s,
        total_s_per_step=step.total_s,
        total_gpus=step.total_gpus,
        total_steps=cost.total_steps,
        total_time_hours=cost.total_time_hours,
        total_cost_usd=cost.total_cost_usd,
        cost_per_1k_tokens_usd=cost.cost_per_1k_tokens_usd,
        power_watts_per_gpu=power_per_gpu,
        total_power_kw=total_power_kw,
        total_energy_kwh=total_energy_kwh,
    )


@router.post("/inference", response_model=InferenceResponse)
def calculate_inference(req: InferenceRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        gpu = get_gpu(req.gpu_id)
        result = simulate_serving(
            model,
            gpu,
            precision=req.precision,
            prompt_tokens=req.prompt_tokens,
            output_tokens=req.output_tokens,
            decode_batch_size=req.decode_batch_size,
            requests_per_sec=req.requests_per_sec,
            cache_hit_fraction=req.cache_hit_fraction,
            utilization=req.utilization,
            paged_attention=req.paged_attention,
            block_size=req.block_size,
            tp_degree=req.tp_degree,
        )
        if result.ttft_s == float("inf") or result.decode_step_s == float("inf"):
            raise ValueError(f"Tensor parallelism (TP={req.tp_degree}) needs NVLink, but {gpu.name} has none.")
        capacity = estimate_serving_capacity(
            model,
            gpu,
            precision=req.precision,
            avg_seq_len=req.prompt_tokens + req.output_tokens,
            gpu_memory_utilization=req.gpu_memory_utilization,
            tp_degree=req.tp_degree,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return InferenceResponse(
        ttft_ms=result.ttft_s * 1000,
        decode_step_ms=result.decode_step_s * 1000,
        tokens_per_sec_per_gpu=result.tokens_per_sec_per_gpu,
        prefill_interference_fraction=result.prefill_interference_fraction,
        colocated_tokens_per_sec_per_gpu=result.colocated_tokens_per_sec_per_gpu,
        disaggregated_tokens_per_sec_per_gpu=result.disaggregated_tokens_per_sec_per_gpu,
        tp_communication_overhead_fraction=result.tp_communication_overhead_fraction,
        tokens_per_sec_per_gpu_amortized=result.tokens_per_sec_per_gpu_amortized,
        usable_vram_gb=capacity.usable_vram_gb,
        weights_gb=capacity.weights_gb,
        kv_budget_gb=capacity.kv_budget_gb,
        max_concurrent_sequences=capacity.max_concurrent_sequences,
    )


@router.post("/recommend", response_model=RecommendResponse)
def calculate_recommend(req: RecommendRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        candidates = recommend_configurations(
            model,
            precision=req.precision,
            tokens_per_step=req.tokens_per_step,
            total_training_tokens=req.total_training_tokens,
            objective=req.objective,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            utilization=req.utilization,
            num_microbatches=req.num_microbatches,
            max_gpus=req.max_gpus,
            budget_usd=req.budget_usd,
            max_time_hours=req.max_time_hours,
            candidate_gpu_ids=req.candidate_gpu_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return RecommendResponse(
        candidates=[RecommendationCandidateOut(**c.__dict__) for c in candidates]
    )


@router.post("/speculative-decoding", response_model=SpeculativeDecodingResponse)
def calculate_speculative_decoding(req: SpeculativeDecodingRequest):
    draft = ModelShape(**req.draft_model.model_dump())
    target = ModelShape(**req.target_model.model_dump())
    try:
        gpu = get_gpu(req.gpu_id)
        result = simulate_speculative_decoding(
            draft,
            target,
            gpu,
            precision=req.precision,
            gamma=req.gamma,
            acceptance_rate=req.acceptance_rate,
            avg_kv_tokens=req.avg_kv_tokens,
            batch_size=req.batch_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return SpeculativeDecodingResponse(
        draft_step_ms=result.draft_step_s * 1000,
        verify_step_ms=result.verify_step_s * 1000,
        round_ms=result.round_s * 1000,
        expected_tokens_per_round=result.expected_tokens_per_round,
        speculative_tokens_per_sec=result.speculative_tokens_per_sec,
        baseline_tokens_per_sec=result.baseline_tokens_per_sec,
        speedup=result.speedup,
    )
