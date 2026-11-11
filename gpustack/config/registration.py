import os
from typing import Optional

registration_token_filename = "token"
worker_token_filename = "worker_token"


def read_token(data_dir: str, filename) -> Optional[str]:
    token_path = os.path.join(data_dir, filename)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    return None


def write_token(data_dir: str, filename: str, token: str):
    token_path = os.path.join(data_dir, filename)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            existing_token = f.read().strip()
        if existing_token == token:
            return  # Token is already written
    with open(token_path, "w") as f:
        f.write(token + "\n")


def read_worker_token(data_dir: str) -> Optional[str]:
    return read_token(data_dir, worker_token_filename)


def write_worker_token(data_dir: str, token: str):
    write_token(data_dir, worker_token_filename, token)


def read_registration_token(data_dir: str) -> Optional[str]:
    return read_token(data_dir, registration_token_filename)


def write_registration_token(data_dir: str, token: str):
    write_token(data_dir, registration_token_filename, token)