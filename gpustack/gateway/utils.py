from typing import List
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.models import ModelInstancePublic
from typing import Union
from dataclasses import dataclass
from typing import Optional
import re
from fastapi import HTTPException
from gpustack.api.exceptions import NotFoundException
from starlette.datastructures import Headers
from typing import Mapping


model_id_prefix = "model-"


router_header_key = "X-GPUStack-Model-Instance"


@dataclass
class RoutePrefix:
    prefixes: List[str]
    support_legacy: bool = False
    additional_versions: Optional[List[str]] = None

    def flattened_prefixes(self) -> List[str]:
        versioned_prefixes = ["/v1"]
        if self.support_legacy:
            versioned_prefixes.append("/v1-openai")
        if self.additional_versions:
            versioned_prefixes.extend(self.additional_versions)
        flattened = []
        for versioned_prefix in versioned_prefixes:
            for prefix in self.prefixes:
                flattened.append(f"{versioned_prefix}{prefix}")
        return flattened

    def regex_prefixes(self) -> List[str]:
        """
        Returns regex patterns for the prefixes, considering versioning and legacy support.
        It supports removing -openai suffix from the versioned prefix with rewrite-target: /$1$3
        """
        versioned_prefixes = [f"/(v1){'(-openai)?' if self.support_legacy else '()'}"]
        if self.additional_versions:
            versioned_prefixes.extend(
                f"/({re.escape(additional_version.lstrip('/'))})()"
                for additional_version in self.additional_versions
            )
        return [
            f"{versioned_prefix}({prefix})"
            for versioned_prefix in versioned_prefixes
            for prefix in self.prefixes
        ]


openai_model_prefixes: List[RoutePrefix] = [
    RoutePrefix(
        [
            "/chat/completions",
            "/completions",
            "/responses",
            "/embeddings",
            "/audio/transcriptions",
            "/audio/speech",
            "/images/generations",
            "/images/edits",
        ],
        True,
    ),
    RoutePrefix(
        [
            "/audio/translations",
            "/images/variations",
            "/moderations",
            "/score",
        ]
    ),
    RoutePrefix(["/rerank"], additional_versions=["/v2"]),
]


def model_prefix(model_id: int) -> str:
    return f"{model_id_prefix}{model_id}-"


def model_instance_prefix(
    model_instance: Union[ModelInstance, ModelInstancePublic]
) -> str:
    return f"{model_prefix(model_instance.model_id)}{model_instance.id}"


def get_instance_id_from_header(headers: Mapping[str, str]) -> int:
    """Parse the model instance ID from the ``x-gpustack-model-instance`` routing header.

    The header value follows the pattern
    ``model-<model_id>-<instance_id>[-l<lora>].<type>`` injected by the API
    gateway. The instance ID is the second numeric segment; LoRA targets append
    an extra ``-l<hash>`` alias segment (see ``lora_registry_name_suffix``) which
    must be skipped.

    Raises:
        HTTPException (400): if the header is absent.
        NotFoundException: if the header value does not match the expected pattern.
    """
    if not isinstance(headers, Headers):
        headers = Headers(headers)
    model_destination = headers.get(router_header_key)
    if model_destination is None:
        raise HTTPException(
            status_code=400, detail=f"Missing {router_header_key} header"
        )

    # Match pattern: model-<model_id>-<instance_id>[-<alias>].<type>
    # instance_id is the second numeric segment, optionally followed by a LoRA alias.
    match = re.match(r'^model-\d+-(\d+)(?:-[^.]+)?\..+', model_destination)
    if not match:
        raise NotFoundException(
            message=f"Invalid model destination format: {model_destination}"
        )

    return int(match.group(1))
