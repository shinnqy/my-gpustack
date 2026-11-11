from typing import Optional
import os
import logging


logger = logging.getLogger(__name__)


legacy_uuid_filename = "worker_uuid"


worker_name_filename = "worker_name"


def get_legacy_uuid(data_dir: str) -> Optional[str]:
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    if os.path.exists(legacy_uuid_path):
        with open(legacy_uuid_path, "r") as file:
            return file.read().strip()
    return None


def set_legacy_uuid(data_dir: str, legacy_uuid: str):
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    with open(legacy_uuid_path, "w") as file:
        file.write(legacy_uuid)


def get_worker_name(data_dir: str) -> Optional[str]:
    worker_name_path = os.path.join(data_dir, worker_name_filename)
    if os.path.exists(worker_name_path):
        with open(worker_name_path, "r") as file:
            return file.read().strip()
    return None


def set_worker_name(data_dir: str, worker_name: str):
    worker_name_path = os.path.join(data_dir, worker_name_filename)
    current_worker_name = get_worker_name(data_dir)
    if current_worker_name is None or current_worker_name != worker_name:
        logger.warning(
            f"Worker name is being updated from {current_worker_name or '<empty>'} to {worker_name}"
        )
        with open(worker_name_path, "w") as file:
            file.write(worker_name)
