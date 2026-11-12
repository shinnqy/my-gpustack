from gpustack.utils.command import find_int_parameter
from typing import List
from typing import Optional
from dataclasses import dataclass

"""整理 vLLM 的并行参数"""

@dataclass
class MultinodeUserParallelism:
    """User-supplied parallelism hints from backend_parameters.

    Any field left as ``None`` means GPUStack should derive a value from the
    physical topology. Non-None values are treated as hard constraints and
    rejected if they contradict the topology.
    """

    tp: Optional[int] = None
    pp: Optional[int] = None
    dp: Optional[int] = None
    dpl: Optional[int] = None
    # Prefill context parallel: multiplies the per-DP-group world like tp/pp
    # (vLLM world_size = pp * tp * pcp). Decode CP is deliberately absent — it
    # reuses the TP group's GPUs and never widens the topology.
    pcp: Optional[int] = None


def parse_user_parallelism(
    backend_parameters: Optional[List[str]],
) -> MultinodeUserParallelism:
    """Extract MultinodeUserParallelism from user-provided backend_parameters."""
    return MultinodeUserParallelism(
        tp=find_int_parameter(backend_parameters, ["tensor-parallel-size", "tp"]),
        pp=find_int_parameter(backend_parameters, ["pipeline-parallel-size", "pp"]),
        dp=find_int_parameter(backend_parameters, ["data-parallel-size", "dp"]),
        dpl=find_int_parameter(backend_parameters, ["data-parallel-size-local", "dpl"]),
        pcp=find_int_parameter(
            backend_parameters, ["prefill-context-parallel-size", "pcp"]
        ),
    )
