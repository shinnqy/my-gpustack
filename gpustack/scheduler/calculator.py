from gpustack.config.config import get_global_config
from gpustack.schemas.models import Model
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker
from gpustack.utils.hub import get_pretrained_config
from gpustack.utils.hub import read_repo_file_content
from gpustack.utils.hub import safe_pretrained_config_from_dict
from typing import Any
from typing import List
from typing import Optional
import asyncio
import os
import logging
from gpustack.schemas.models import BackendEnum
from gpustack.schemas.models import is_audio_model
from typing import Dict
import json


logger = logging.getLogger(__name__)


async def get_pretrained_config_with_workers(
    model: Model,
    workers: Optional[List[Worker]] = None,
    trust_remote_code: bool = False,
) -> Optional[Any]:
    """
    调度阶段获取配置的统一入口

    Server 看不到模型目录
        ↓
    get_pretrained_config_from_workers()
        ↓
    read_local_path_file_from_workers()
    """

    pretrained_config = None
    timeout_in_seconds = 15

    try:
        if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(
            model.local_path
        ):
            logger.info(
                f"Model path '{model.local_path}' does not exist in server node. Trying workers..."
            )
            """如果模型来源是本地目录，而 Server 看不到这个目录，就转到 Worker 读取分支"""
            pretrained_config = await get_pretrained_config_from_workers(
                model,
                workers,
            )
        else:
            logger.info(f"Trying to get pretrained config from {model.source}")
            pretrained_config = await asyncio.wait_for(
                asyncio.to_thread(
                    get_pretrained_config, model, trust_remote_code=trust_remote_code
                ),
                timeout=timeout_in_seconds,
            )
    except Exception as e:
        if should_fallback_load_config_json(e, model):
            config_dict = await asyncio.wait_for(
                asyncio.to_thread(
                    read_repo_file_content,
                    model,
                    "config.json",
                    token=get_global_config().huggingface_token,
                ),
                timeout=timeout_in_seconds,
            )
            if config_dict:
                return safe_pretrained_config_from_dict(config_dict)

        raise e

    return pretrained_config


def should_fallback_load_config_json(e: Exception, model: Model) -> bool:
    """
    配置加载失败时，判断能否直接读取 config.json
    """

    # For LOCAL_PATH models, the path must be a valid directory
    if model.source == SourceEnum.LOCAL_PATH and not (
        model.local_path and os.path.isdir(model.local_path)
    ):
        return False

    if model.backend == BackendEnum.VLLM and is_audio_model(model):
        # TODO(michelia): Qwen3-ASR is currently supported by vLLM but not yet by Hugging Face Transformers.
        # Track upstream progress: https://github.com/huggingface/transformers/issues/43837
        return True

    # For very new HF checkpoints, AutoConfig may fail on an older transformers.
    # Falling back to reading config.json avoids requiring a transformers upgrade.
    #
    # Example upstream error message (may vary by transformers version):
    """
    The checkpoint you are trying to load has model type `{config_dict['model_type']}`
    but Transformers does not recognize this architecture. This could be because of an
    issue with the checkpoint, or because your version of Transformers is out of date.
    You can update Transformers with the command `pip install --upgrade transformers`.
    If this does not work, and the checkpoint is very new, then there may not be a
    release version that supports this model yet. In this case, you can get the most
    up-to-date code by installing Transformers from source with the command
    `pip install git+https://github.com/huggingface/transformers.git`
    """
    msg = str(e).lower()
    if (
        "update transformers" in msg
        or "pip install --upgrade transformers" in msg
        or "does not recognize this architecture" in msg
        or "install transformers from source" in msg
    ):
        return True

    # Fallback for backend version specified or import errors
    return model.backend_version is not None or isinstance(e, ImportError)


async def read_local_path_file_from_workers(  # noqa: C901
    model: Model,
    file_path: str,
    workers: List[Worker],
) -> Optional[Dict[str, Any]]:
    """
    Read a file from LOCAL_PATH model by querying workers.

    Steps:
    1. Apply filters (GPU selector, label selector) to reduce broadcast scope
    2. Broadcast to filtered workers

    Args:
        model: Model with source LOCAL_PATH
        file_path: Relative path to the file (e.g., "config.json", "model_index.json")
        workers: List of workers to query

    Returns:
        File content as dict if successful

    Raises:
        ValueError: If the file cannot be read from any worker, with per-worker diagnostic details.
    """
    if not workers:
        raise ValueError(
            f"No workers are available to read '{file_path}' for model '{model.name}'. "
        )
    raise ValueError(
        f"Cannot read '{file_path}' from workers yet. Make model path '{model.local_path}' "
        "accessible on the Server or use a Hugging Face/ModelScope repository."
    )


async def get_pretrained_config_from_workers(
    model: Model,
    workers: List[Worker],
) -> Optional[Any]:
    """
    如果模型来源是本地目录，而 Server 看不到这个目录，就转到 Worker 读取分支
    """
    config_dict = await read_local_path_file_from_workers(model, "config.json", workers)

    return safe_pretrained_config_from_dict(config_dict)


def read_model_index_json(path: str) -> dict:
    """
    Read and parse model_index.json from local directory.

    Args:
        path: Directory path containing model_index.json

    Returns:
        Parsed JSON data from model_index.json, or None if not found

    Raises:
        json.JSONDecodeError: If model_index.json is invalid
        PermissionError: If permission denied
        OSError: For other I/O errors
    """
    model_index_path = os.path.join(path, "model_index.json")

    if not os.path.exists(model_index_path):
        return None

    try:
        with open(model_index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except PermissionError:
        logger.error(f"Permission denied reading model_index.json: {model_index_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse model_index.json: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to read model_index.json: {e}")
        raise


def calculate_llm_model_weight_size(path: str) -> int:
    """
    普通模型统计 Server 本地权重文件大小
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    total_size = 0

    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(weight_file_extensions):
                    total_size += entry.stat().st_size
    except PermissionError:
        logger.error(f"Permission denied when accessing '{path}'.")
        raise

    return total_size


def calculate_diffusion_model_weight_size(path: str) -> int:
    """
    diffusion 模型统计 Server 本地权重文件大小
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path '{path}' is not a directory.")

    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")

    # Read pipeline definition
    pipeline_data = read_model_index_json(path)

    if pipeline_data is None:
        raise FileNotFoundError(f"model_index.json not found in {path}")

    if not isinstance(pipeline_data, dict):
        raise TypeError(f"model_index.json in {path} is not a valid JSON object.")

    # Remove metadata keys (starting with _)
    component_dirs = {key for key in pipeline_data.keys() if not key.startswith('_')}

    total_size = 0

    # Scan each component directory
    for component_dir in component_dirs:
        component_path = os.path.join(path, component_dir)

        if not os.path.isdir(component_path):
            continue

        # Scan files in component directory
        try:
            with os.scandir(component_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(weight_file_extensions):
                        total_size += entry.stat().st_size
        except PermissionError:
            logger.error(f"Permission denied scanning directory: {component_path}")
            raise
        except OSError as e:
            logger.error(f"Error scanning directory {component_path}: {e}")
            raise

    return total_size


def calculate_local_model_weight_size(path: str, is_diffusion: bool = False) -> int:
    """
    Calculate model weight size based on model type.

    Unified entry point for calculating model weight sizes.

    Args:
        path: Directory path to scan
        is_diffusion: Whether this is a diffusion model (default: False)

    Returns:
        Total size in bytes of weight files

    Raises:
        FileNotFoundError: If path doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If permission denied
        json.JSONDecodeError: If model_index.json is invalid (diffusion only)
    """
    if is_diffusion:
        return calculate_diffusion_model_weight_size(path)
    else:
        return calculate_llm_model_weight_size(path)
