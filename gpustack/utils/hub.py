from gpustack.schemas.models import Model
from gpustack.schemas.models import SourceEnum
from typing import Optional
from gpustack.utils.cache import load_cache
from gpustack.utils.cache import save_cache
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id
from modelscope.hub.api import HubApi
from pathlib import Path
from typing import Dict
from typing import List
import json
from gpustack.config.config import get_global_config
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
import os
from typing import Any
import gzip
import tempfile
from transformers import PretrainedConfig
from gpustack.schemas.models import CategoryEnum
import logging
from functools import cache
from threading import Lock


logger = logging.getLogger(__name__)


LIST_REPO_CACHE_DIR = "repo-skeleton"


MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN = [
    "config.json",
    "*.py",
]


@cache
def get_model_lock(model_id: str) -> Lock:
    """同一进程内，同一模型共用一个线程锁，避免多个调度评估线程同时下载到同一个目录。进入锁后再次检查 config.json，防止重复下载"""
    return Lock()


def list_repo(
    repo_id: str,
    source: str,
    token: Optional[str] = None,
    cache_expiration: Optional[int] = None,
    root_dir_only: bool = False,
) -> List[Dict[str, any]]:
    """
    统一获取仓库文件清单
        - Hugging Face 和 ModelScope 的接口、返回字段不同，它把结果整理成统一格式
    """
    cache_key = f"{source}:{repo_id}:{root_dir_only}"
    cached_result, is_succ = load_cache(
        LIST_REPO_CACHE_DIR, cache_key, cache_expiration
    )
    if is_succ:
        result = json.loads(cached_result)
        if isinstance(result, list):
            return result

    if source == SourceEnum.HUGGING_FACE:
        validate_repo_id(repo_id)
        hffs = HfFileSystem(token=token)
        file_info = []
        for file in hffs.ls(repo_id, recursive=not root_dir_only):
            if not isinstance(file, dict):
                continue
            relative_path = Path(file["name"]).relative_to(repo_id).as_posix()
            # If root_only is True, skip files in subdirectories
            if root_dir_only and "/" in relative_path:
                continue
            file_info.append(
                {
                    "name": relative_path,
                    "size": file["size"],
                }
            )
    elif source == SourceEnum.MODEL_SCOPE:
        msapi = HubApi()
        files = msapi.get_model_files(repo_id, recursive=not root_dir_only)
        file_info = []
        for file in files:
            file_path = file["Path"]
            # If root_only is True, skip files in subdirectories
            if root_dir_only and "/" in file_path:
                continue
            file_info.append(
                {
                    "name": file_path,
                    "size": file["Size"],
                }
            )
    else:
        raise ValueError(f"Invalid source: {source}")

    if not save_cache(LIST_REPO_CACHE_DIR, cache_key, json.dumps(file_info)):
        logger.info(f"Saved cache {LIST_REPO_CACHE_DIR} {cache_key} fail")

    return file_info


def read_repo_file_content(  # noqa: C901
    model: Model,
    file_path: str,
    token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    读取指定的 JSON 文件 (Hugging Face / ModelScope / 本地目录)
    """
    try:
        if model.source == SourceEnum.HUGGING_FACE:
            hffs = HfFileSystem(token=token)
            repo_path = f"{model.huggingface_repo_id}/{file_path}"
            with hffs.open(repo_path, "rb") as f:
                content = f.read()
                if (
                    content
                    and content.startswith(b"\x1f\x8b")
                    and not file_path.endswith(".gz")
                ):
                    try:
                        content = gzip.decompress(content)
                    except Exception as e:
                        logger.warning(
                            f"Failed to decompress gzip content for {file_path}: {e}"
                        )
                return json.loads(content)

        elif model.source == SourceEnum.MODEL_SCOPE:
            _cfg = get_global_config()
            base_tmp = os.path.join(
                (_cfg.cache_dir if _cfg and _cfg.cache_dir else "/tmp"),
                "modelscope",
                "tempfile",
            )
            os.makedirs(base_tmp, exist_ok=True)
            safe_id = (model.model_scope_model_id or "").replace("/", "__")
            with tempfile.TemporaryDirectory(
                dir=base_tmp, prefix=f"{safe_id}__"
            ) as tmp_dir:
                model_dir = modelscope_snapshot_download(
                    model_id=model.model_scope_model_id,
                    local_dir=tmp_dir,
                    allow_patterns=[file_path],
                )

                candidate = os.path.join(model_dir, file_path)
                fp = candidate if os.path.exists(candidate) else None
                if not fp:
                    # Search recursively by base filename for robustness
                    base_name = os.path.basename(file_path)
                    for root, _dirs, files in os.walk(model_dir):
                        if base_name in files:
                            fp = os.path.join(root, base_name)
                            break
                if not fp:
                    return None
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)

        elif model.source == SourceEnum.LOCAL_PATH:
            local_path = model.local_path or ""
            if not local_path or not os.path.isdir(local_path):
                return None
            fp = os.path.join(local_path, file_path)
            if os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    return json.load(f)

            return None
        else:
            return None
    except Exception as e:
        source_key = (
            model.huggingface_repo_id
            or model.model_scope_model_id
            or model.local_path
            or "<unknown>"
        )
        logger.error(f"Failed to read '{file_path}' for source '{source_key}': {e}")
        return None


def get_model_weight_size(model: Model, token: Optional[str] = None) -> int:
    """
    从文件清单中统计权重大小
        - 把符合条件的文件大小相加，同时排除 consolidated.safetensors，避免某些仓库里完整权重与分片权重重复计算
    """
    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    # consolidated.safetensors is usually a duplicate of other weight files. Exclude by default.
    # Example: https://huggingface.co/mistralai/Voxtral-Small-24B-2507
    exclude_files = ["consolidated.safetensors"]
    if model.source == SourceEnum.HUGGING_FACE:
        repo_id = model.huggingface_repo_id
    elif model.source == SourceEnum.MODEL_SCOPE:
        repo_id = model.model_scope_model_id
    else:
        raise ValueError(f"Unknown source {model.source}")
    repo_file_infos = list_repo(repo_id, model.source, token=token, root_dir_only=True)
    return sum(
        file.get("size", 0)
        for file in repo_file_infos
        if (
            file.get("name", "").endswith(weight_file_extensions)
            and file.get("name", "") not in exclude_files
        )
    )


def get_diffusion_model_weight_size(model: Model, token: Optional[str] = None) -> int:
    """
    先读取 model_index.json，再统计其中声明的组件目录下的权重，例如 transformer/、`vae/`、`text_encoder/`
    """
    weight_file_extensions = (".safetensors", ".bin", ".pt", ".pth")
    if model.source == SourceEnum.HUGGING_FACE:
        repo_id = model.huggingface_repo_id
    elif model.source == SourceEnum.MODEL_SCOPE:
        repo_id = model.model_scope_model_id
    else:
        raise ValueError(f"Unknown source {model.source}")
    if not model.categories or CategoryEnum.IMAGE not in model.categories:
        raise ValueError("Model is not an image model")

    # In different repositories, model files may be stored in different dir.
    # However, during runtime, the diffusers loads components from corresponding dir according to the pipeline defined in model_index.json.
    # We can follow the definition in model_index.json to determine which file weights should be included in the calculation.
    pipeline_data = read_repo_file_content(model, "model_index.json", token=token)
    if pipeline_data is None:
        raise ValueError(f"No model_index.json in repo {repo_id}")
    if isinstance(pipeline_data, list) and len(pipeline_data) > 0:
        pipeline_data = pipeline_data[0]

    sum_size = 0
    repo_file_infos = list_repo(repo_id, model.source, token=token, root_dir_only=False)
    for file_info in repo_file_infos:
        name_split = file_info.get("name", "").split("/", 1)
        if (
            len(name_split) <= 1
            or pipeline_data.get(name_split[0], None) is None
            or not name_split[1].endswith(weight_file_extensions)
        ):
            continue
        sum_size += file_info.get("size", 0)

    return sum_size


def get_pretrained_config(model: Model, **kwargs):
    """
    把配置加载成 Transformers 配置对象
    """

    trust_remote_code = False
    if (
        model.backend_parameters and "--trust-remote-code" in model.backend_parameters
    ) or kwargs.get("trust_remote_code"):
        trust_remote_code = True

    global_config = get_global_config()
    pretrained_config = None
    if model.source == SourceEnum.HUGGING_FACE:
        from transformers import AutoConfig

        pretrained_config = AutoConfig.from_pretrained(
            model.huggingface_repo_id,
            token=global_config.huggingface_token,
            trust_remote_code=trust_remote_code,
            cache_dir=os.path.join(global_config.cache_dir, "huggingface"),
        )
    elif model.source == SourceEnum.MODEL_SCOPE:
        from modelscope import AutoConfig

        model_scope_cache_dir = os.path.join(global_config.cache_dir, "model_scope")
        repo_cache_dir = os.path.join(
            model_scope_cache_dir, *model.model_scope_model_id.split('/')
        )
        ms_config_json = os.path.join(repo_cache_dir, "config.json")
        # ModelScope's wrapped AutoConfig passes kwargs to HF from_pretrained after
        # snapshot_download; cache_dir/local_dir are not forwarded to snapshot_download,
        # so downloads must use modelscope_snapshot_download with local_dir here.
        # Require config.json (not merely an existing dir) so empty directories still fetch.
        if not os.path.isfile(ms_config_json):
            logger.info(
                f"Downloading ModelScope files required for config to {repo_cache_dir} "
                f"(model_id={model.model_scope_model_id})"
            )
            with get_model_lock(model.model_scope_model_id):
                if not os.path.isfile(ms_config_json):
                    modelscope_snapshot_download(
                        model_id=model.model_scope_model_id,
                        local_dir=repo_cache_dir,
                        allow_file_pattern=MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN,
                    )
        else:
            logger.info(f"Repo cache dir: {repo_cache_dir}")

        logger.info(
            f"Loading pretrained config for ModelScope model {model.model_scope_model_id} "
            f"from {repo_cache_dir}"
        )
        pretrained_config = AutoConfig.from_pretrained(
            repo_cache_dir,
            trust_remote_code=trust_remote_code,
            allow_file_pattern=MODELSCOPE_CONFIG_ALLOW_FILE_PATTERN,
            local_files_only=True,
        )
        logger.info(
            f"Successfully loaded pretrained config for ModelScope model {model.model_scope_model_id}"
        )
    elif model.source == SourceEnum.LOCAL_PATH:
        if not os.path.exists(model.local_path):
            logger.warning(
                f"Local Path: {model.readable_source} is not local to the server node and may reside on a worker node."
            )
            # Return an empty dict here to facilitate special handling by upstream methods.
            return {}

        from transformers import AutoConfig

        pretrained_config = AutoConfig.from_pretrained(
            model.local_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    else:
        raise ValueError(f"Unsupported model source: {model.source}")

    return pretrained_config


def safe_pretrained_config_from_dict(config_dict: Dict) -> PretrainedConfig:
    """
    从原始 JSON 重建配置对象。暂时移除旧式 rope_scaling 字段，构造完成后再恢复，避开初始化兼容问题。
    """
    config_dict = dict(config_dict)
    rope_scaling = config_dict.pop("rope_scaling", None)
    config = PretrainedConfig.from_dict(config_dict)
    if rope_scaling is not None:
        config.rope_scaling = rope_scaling
    return config


def get_max_model_len(pretrained_config) -> int:  # noqa: C901
    """
    从不同模型使用的长度字段中推导上下文长度；多个字段同时存在时取最小值，全部缺失时使用 2048，再按部分 RoPE 配置修正。
    """
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(pretrained_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(pretrained_config, "rope_scaling", None)
    if rope_scaling is not None and isinstance(rope_scaling, dict):
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            # Per-attention-type rope_parameters (e.g., Gemma-4) surface as a
            # nested dict keyed by attention type with no top-level rope_type.
            # No single scaling factor applies; fall through to derived length.
            rope_type = None

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type is not None and rope_type not in ("su", "longrope", "llama3"):
            scaling_factor = 1
            if "factor" in rope_scaling:
                scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    logger.debug(f"Derived max model length: {derived_max_model_len}")
    return int(derived_max_model_len)


def get_hf_text_config(config: PretrainedConfig):
    """
    对多模态模型提取内部的文本配置，让调度器读取语言模型的注意力头数等参数。
    """
    if hasattr(config, "text_config"):
        text_config = config.get_text_config()
        if text_config is not None:
            if isinstance(text_config, dict):
                text_config = safe_pretrained_config_from_dict(text_config)
            if hasattr(text_config, "num_attention_heads"):
                return text_config

    return config
