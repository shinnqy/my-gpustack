from typing import Tuple
import re


pattern = r"^(?P<worker_name>.+):(?P<device>[^:]+):(?P<gpu_index>\d+)$"


def parse_gpu_id(input: str) -> Tuple[bool, dict]:
    """
    Parse the input string to check if it matches the format worker_name:device:gpu_index.

    Args:
        input_string (str): The input string to parse.

    Returns:
        tuple: (bool, dict)
               - If matched, the first value is True, and the second value is a dictionary
                 containing worker_name, device, and gpu_index.
               - If not matched, the first value is False, and the second value is None.
    """

    match = re.match(pattern, input)
    if match:
        return True, match.groupdict()
    return False, None


def group_gpu_ids_by_worker(gpu_ids: list) -> dict:
    """
    Group GPU IDs by worker name.

    Args:
        gpu_ids (list): List of GPU IDs.

    Returns:
        dict: A dictionary where the keys are worker names and the values are lists of GPU IDs.
    """

    worker_gpu_ids = {}
    for gpu_id in gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise ValueError(f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        if worker_name not in worker_gpu_ids:
            worker_gpu_ids[worker_name] = []
        worker_gpu_ids[worker_name].append(gpu_id)

    for worker_name, gpu_ids in worker_gpu_ids.items():
        worker_gpu_ids[worker_name] = sorted(gpu_ids)
    return worker_gpu_ids


def group_gpu_indexes_by_gpu_type_and_worker(gpu_ids: list) -> dict:
    """
    整理手动指定的 GPU:
        - 把用户选中的显卡，按“设备类型 → 机器 → 卡号”整理好，方便调度器查找
    """
    result = {}
    for gpu_id in gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise ValueError(f"Invalid GPU ID: {gpu_id}")
        worker = matched["worker_name"]
        gpu_type = matched["device"]
        gpu_index = int(matched["gpu_index"])
        result.setdefault(gpu_type, {}).setdefault(worker, []).append(gpu_index)
    # Sort indexes for each type/worker
    return {
        t: {w: sorted(idx) for w, idx in workers.items()}
        for t, workers in result.items()
    }
