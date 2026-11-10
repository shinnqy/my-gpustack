import argparse
import asyncio
import logging
import multiprocessing
import sys
from typing import Any, Dict

import yaml

from gpustack import __version__, __git_commit__
from gpustack.config.config import set_global_config
from gpustack.logging import setup_logging
from gpustack.utils.envs import get_gpustack_env, get_gpustack_env_bool
from gpustack.worker.worker import Worker
from gpustack.config import Config
from gpustack.server.server import Server

logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = kwargs.get("default")

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run GPUStack server or worker.",
        description="Run GPUStack server or worker.",
    )
    start_cmd_options(parser_server)


def start_cmd_options(parser_server: argparse.ArgumentParser):
    common_group = parser_server.add_argument_group("Common settings")
    common_group.add_argument(
        "--api-port",
        type=int,
        help="Port to bind the GPUStack API server to.",
        default=get_gpustack_env("API_PORT"),
    )
    common_group.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML config file.",
        default=get_gpustack_env("CONFIG_FILE"),
    )
    common_group.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=get_gpustack_env_bool("DEBUG"),
    )
    common_group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data. The default is OS specific.",
        default=get_gpustack_env("DATA_DIR"),
    )
    server_group = parser_server.add_argument_group("Server settings")
    server_group.add_argument(
        "--database-url",
        type=str,
        help="URL of the database. Example: postgresql://user:password@hostname:port/db_name.",
        default=get_gpustack_env("DATABASE_URL"),
    )
    server_group.add_argument(
        "--enable-worker",
        action=OptionalBoolAction,
        help="Enable the embedded worker for the GPUStack server.",
        default=get_gpustack_env_bool("ENABLE_WORKER"),
    )
    worker_group = parser_server.add_argument_group("Worker settings")
    worker_group.add_argument(
        "-t",
        "--token",
        type=str,
        help="Shared secret used to add a worker.",
        default=get_gpustack_env("TOKEN"),
    )
    worker_group.add_argument(
        "-s",
        "--server-url",
        type=str,
        help="Server to connect to.",
        default=get_gpustack_env("SERVER_URL"),
    )
    worker_group.add_argument(
        "--worker-ip",
        type=str,
        help="IP address of the worker node. Auto-detected by default.",
        default=get_gpustack_env("WORKER_IP"),
    )
    worker_group.add_argument(
        "--worker-name",
        type=str,
        help="Name of the worker node. Use the hostname by default.",
        default=get_gpustack_env("WORKER_NAME"),
    )
    worker_group.add_argument(
        "--worker-port",
        type=int,
        help="Port to bind the worker to.",
        default=get_gpustack_env("WORKER_PORT"),
    )
    parser_server.set_defaults(func=run)

def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        multiprocessing.set_start_method('spawn')

        logger.info(f"GPUStack version: {__version__} ({__git_commit__})")

        if cfg.server_url:
            run_worker(cfg)
        else:
            run_server(cfg)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)

def run_server(cfg: Config):
    server = Server(
        config=cfg,
        # spawn 模式下，`cfg` 会被 pickle 传给子进程。子进程的全局变量是空的，所以 `run_worker` 第一行要再 `set_global_config(cfg)`
        worker_process=multiprocessing.Process(target=run_worker, args=(cfg,)),
    )

    try:
        # Server 整体是异步的（数据库、HTTP、后台循环都是协程）
        asyncio.run(server.start())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        raise
    finally:
        logger.info("Server has shut down.")

def run_worker(cfg: Config):
    set_global_config(cfg)
    worker = Worker(cfg)
    # `Worker.start()` 是同步的，它在内部自己 `asyncio.run`，因为它启动前要做一些同步准备
    worker.start()

def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_args(args: argparse.Namespace) -> Config:
    config_data = {}
    if args.config_file:
        config_data.update(load_config_from_yaml(args.config_file))

    # CLI args have higher priority than config file
    set_common_options(args, config_data)
    set_server_options(args, config_data)
    set_worker_options(args, config_data)

    try:
        cfg = Config(**config_data)
    except Exception as e:
        raise Exception(f"Config error: {e}")

    set_global_config(cfg)

    return cfg


def set_config_option(args, config_data: dict, option_name: str):
    option_value = getattr(args, option_name, None)
    if option_value is not None:
        config_data[option_name] = option_value


def set_common_options(args, config_data: dict):
    options = [
        "debug",
        "data_dir",
        "api_port",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_server_options(args, config_data: dict):
    options = [
        "database_url",
        "enable_worker",
    ]

    for option in options:
        set_config_option(args, config_data, option)


def set_worker_options(args, config_data: dict):
    options = [
        "token",
        "server_url",
        "worker_ip",
        "worker_name",
        "worker_port",
    ]

    for option in options:
        set_config_option(args, config_data, option)