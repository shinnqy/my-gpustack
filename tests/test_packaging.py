from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from zipfile import ZipFile

import pytest

from gpustack.worker.tools_manager import FASTFETCH_VERSION, ToolsManager


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def tool_server():
    requests = []
    responses = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = responses.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.end_headers()
            if body is not None:
                self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def archive(member, content=b"native-tool-fixture"):
    data = BytesIO()
    with ZipFile(data, "w") as zipped:
        zipped.writestr(member, content)
    return data.getvalue()


@pytest.mark.parametrize("system,arch,platform,member,filename", [
    ("linux", "arm64", "linux-aarch64", "fastfetch-linux-aarch64/usr/bin/fastfetch", "fastfetch"),
    ("linux", "x86_64", "linux-amd64", "fastfetch-linux-amd64/usr/bin/fastfetch", "fastfetch"),
    ("darwin", "arm64", "macos-universal", "fastfetch-macos-universal/usr/bin/fastfetch", "fastfetch"),
    ("windows", "amd64", "windows-amd64", "fastfetch.exe", "fastfetch.exe"),
])
def test_prepare_tools_downloads_selected_platform_and_reuses_cache(tmp_path, system, arch, platform, member, filename):
    with tool_server() as (base, responses, requests):
        path = f"/gpustack/fastfetch/releases/download/{FASTFETCH_VERSION}/fastfetch-{platform}.zip"
        responses[path] = archive(member)
        manager = ToolsManager(base, system=system, arch=arch, bin_dir=tmp_path / "third_party/bin")
        result = manager.prepare_tools()
        assert result.name == filename
        assert result.read_bytes() == b"native-tool-fixture"
        assert os.access(result, os.X_OK)
        assert (result.parent / "__init__.py").is_file()
        assert manager.prepare_tools() == result
        assert requests == [path]
        metadata = json.loads((result.parent.parent / "versions.json").read_text())
        assert metadata["fastfetch"] == FASTFETCH_VERSION
        assert metadata["fastfetch-platform"] == platform


def test_cross_architecture_tool_cache_is_not_reused(tmp_path):
    with tool_server() as (base, responses, requests):
        for platform in ("linux-amd64", "linux-aarch64"):
            path = f"/gpustack/fastfetch/releases/download/{FASTFETCH_VERSION}/fastfetch-{platform}.zip"
            responses[path] = archive(f"fastfetch-{platform}/usr/bin/fastfetch", platform.encode())
        first = ToolsManager(base, system="linux", arch="amd64", bin_dir=tmp_path / "bin").prepare_tools()
        assert first.read_bytes() == b"linux-amd64"
        second = ToolsManager(base, system="linux", arch="arm64", bin_dir=tmp_path / "bin").prepare_tools()
        assert second.read_bytes() == b"linux-aarch64"
        assert len(requests) == 2


@pytest.mark.parametrize("body", [b"not a ZIP archive", archive("unexpected-file")])
def test_failed_tool_download_preserves_existing_binary(tmp_path, body):
    manager = ToolsManager(system="linux", arch="arm64", bin_dir=tmp_path / "bin")
    target = tmp_path / "bin/fastfetch/fastfetch"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing-tool")
    with tool_server() as (base, responses, _):
        manager._download_base_url = base
        responses[f"/gpustack/fastfetch/releases/download/{FASTFETCH_VERSION}/fastfetch-linux-aarch64.zip"] = body
        with pytest.raises(RuntimeError, match="Failed to download fastfetch"):
            manager.prepare_tools()
    assert target.read_bytes() == b"existing-tool"
    assert not (tmp_path / "bin/versions.json").exists()


def test_unified_image_entrypoint_preserves_worker_arguments(tmp_path):
    executable = tmp_path / "gpustack"
    executable.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "pack/entrypoint.sh"), "--server-url", "http://server:30080",
         "--worker-name", "worker with spaces", "--debug"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == [
        "start", "--server-url", "http://server:30080", "--worker-name", "worker with spaces", "--debug",
    ]


@pytest.mark.parametrize("push,export", [("false", "--load"), ("true", "--push")])
@pytest.mark.parametrize("network", ["default", "host"])
def test_package_buildx_contract(tmp_path, push, export, network):
    executable = tmp_path / "docker"
    log = tmp_path / "docker.jsonl"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['PACKAGE_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "hack/package.sh")], capture_output=True, text=True,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
             "PACKAGE_TEST_LOG": str(log), "PACKAGE_ARCH": "aarch64",
             "PACKAGE_NAMESPACE": "local", "PACKAGE_REPOSITORY": "my-gpustack",
             "PACKAGE_TAG": "chapter15", "PACKAGE_PUSH": push, "PACKAGE_PULL": "false",
             "PACKAGE_NETWORK": network,
             "GPUSTACK_OPERATOR_VERSION": "v0.9.2", "GPUSTACK_BASE_IMAGE": "base",
             "GPUSTACK_TOOLS_DOWNLOAD_BASE_URL": "https://tools.example"},
    )
    assert result.returncode == 0, result.stderr
    command = [json.loads(line) for line in log.read_text().splitlines()][-1]
    assert command[:2] == ["buildx", "build"]
    assert command[command.index("--platform") + 1] == "linux/arm64"
    assert command[command.index("--target") + 1] == "gpustack"
    assert command[command.index("--tag") + 1] == "local/my-gpustack:chapter15"
    assert command[command.index("--file") + 1] == str(ROOT / "pack/Dockerfile")
    assert export in command and ("--push" if push == "false" else "--load") not in command
    assert "--pull" not in command
    assert command[command.index("--network") + 1] == network
    if network == "host":
        assert command[command.index("--allow") + 1] == "network.host"
    else:
        assert "--allow" not in command
    assert "GPUSTACK_OPERATOR_VERSION=v0.9.2" in command
    assert "GPUSTACK_TOOLS_DOWNLOAD_BASE_URL=https://tools.example" in command
    assert command[-1] == str(ROOT)


def test_package_rejects_unsupported_architecture():
    result = subprocess.run(
        ["bash", str(ROOT / "hack/package.sh")], capture_output=True, text=True,
        env={**os.environ, "PACKAGE_ARCH": "s390x"},
    )
    assert result.returncode != 0 and "Unsupported PACKAGE_ARCH" in result.stderr


def test_package_rejects_unsupported_network():
    result = subprocess.run(
        ["bash", str(ROOT / "hack/package.sh")], capture_output=True, text=True,
        env={**os.environ, "PACKAGE_ARCH": "arm64", "PACKAGE_NETWORK": "invalid"},
    )
    assert result.returncode != 0 and "Unsupported PACKAGE_NETWORK" in result.stderr
