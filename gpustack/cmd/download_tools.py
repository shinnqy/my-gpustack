import argparse
import sys

from gpustack.utils.envs import get_gpustack_env
from gpustack.worker.tools_manager import ToolsManager


def setup_download_tools_cmd(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("download-tools", help="Download the dependency tools used by this stage")
    parser.add_argument("--tools-download-base-url", default=get_gpustack_env("TOOLS_DOWNLOAD_BASE_URL"))
    parser.add_argument("--system", default=get_gpustack_env("SYSTEM"))
    parser.add_argument("--arch", default=get_gpustack_env("ARCH"))
    parser.set_defaults(func=run)


def run(args):
    try:
        target = ToolsManager(
            tools_download_base_url=args.tools_download_base_url,
            system=args.system, arch=args.arch,
        ).prepare_tools()
        print(f"Prepared fastfetch: {target}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to prepare dependency tools: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
