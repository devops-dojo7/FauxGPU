from fastapi import APIRouter, HTTPException

from api.schemas import TopologyRequest, TopologyResponse
from engine.topology import build_topology

router = APIRouter(prefix="/topology", tags=["topology"])


@router.post("", response_model=TopologyResponse)
def resolve_topology(req: TopologyRequest):
    try:
        topo = build_topology(
            shape=req.shape,
            gpu_id=req.gpu_id,
            gpus_per_node=req.gpus_per_node,
            num_nodes=req.num_nodes,
            fabric_id=req.fabric_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return TopologyResponse(
        shape=topo.shape,
        gpu_id=topo.gpu.id,
        gpu_name=topo.gpu.name,
        total_gpus=topo.total_gpus,
        intra_node_bandwidth_gbps=topo.intra_node_bandwidth_gbps,
        inter_node_bandwidth_gbps=topo.inter_node_bandwidth_gbps,
        bottleneck_bandwidth_gbps=topo.bottleneck_bandwidth_gbps(),
    )
