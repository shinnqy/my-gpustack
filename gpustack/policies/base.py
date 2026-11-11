from dataclasses import dataclass
import logging
from typing import Optional
from gpustack.schemas.models import (
    ModelInstance,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInstanceScore:
    model_instance: ModelInstance
    score: Optional[float] = None