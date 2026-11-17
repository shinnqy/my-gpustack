"""The fastfetch part of GPUStack's prebuilt-tool preparation."""

import importlib
import json
import os
from pathlib import Path
import platform
import tempfile
import urllib.request
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from gpustack.utils.compat_importlib import pkg_resources


FASTFETCH_VERSION = "2.25.0.1"
TOOLS_DOWNLOAD_BASE_URLS = (
    "https://github.com",
    "https://gpustack-1303613262.cos.ap-guangzhou.myqcloud.com",
)


class ToolsManager:
    def __init__(self, tools_download_base_url=None, system=None, arch=None, bin_dir=None):
        self._download_base_url = tools_download_base_url.rstrip("/") if tools_download_base_url else None
        self._os = (system or platform.system()).lower()
        machine = (arch or platform.machine()).lower()
        self._arch = {"aarch64": "arm64", "x86_64": "amd64"}.get(machine, machine)
        self.third_party_bin_path = (
            Path(bin_dir) if bin_dir is not None else
            Path(str(pkg_resources.files("gpustack.third_party"))) / "bin"
        )

    def _get_fastfetch_platform_name(self):
        if self._os == "darwin":
            return "macos-universal"
        if self._os == "linux" and self._arch in {"amd64", "arm64"}:
            return "linux-amd64" if self._arch == "amd64" else "linux-aarch64"
        if self._os == "windows" and self._arch == "amd64":
            return "windows-amd64"
        raise ValueError(f"Unsupported fastfetch platform: {self._os}/{self._arch}")

    def prepare_tools(self):
        # The later backends can add their own original ToolsManager tools here.
        return self.download_fastfetch()

    def download_fastfetch(self):
        platform_name = self._get_fastfetch_platform_name()
        filename = "fastfetch.exe" if self._os == "windows" else "fastfetch"
        target = self.third_party_bin_path / "fastfetch" / filename
        versions_path = self.third_party_bin_path / "versions.json"
        try:
            versions = json.loads(versions_path.read_text())
        except (FileNotFoundError, ValueError):
            versions = {}
        if not isinstance(versions, dict):
            versions = {}
        if (target.is_file() and versions.get("fastfetch") == FASTFETCH_VERSION
                and versions.get("fastfetch-platform") == platform_name):
            return target
        relative = (
            f"gpustack/fastfetch/releases/download/{FASTFETCH_VERSION}/"
            f"fastfetch-{platform_name}.zip"
        )
        member = "fastfetch.exe" if self._os == "windows" else f"fastfetch-{platform_name}/usr/bin/fastfetch"
        bases = (self._download_base_url,) if self._download_base_url else TOOLS_DOWNLOAD_BASE_URLS
        failures = []
        for base in bases:
            try:
                with urllib.request.urlopen(f"{base}/{relative}", timeout=30) as response:
                    archive = response.read()
                with ZipFile(BytesIO(archive)) as zipped:
                    binary = zipped.read(member)
                if not binary:
                    raise ValueError("fastfetch archive contains an empty executable")
                break
            except (OSError, KeyError, BadZipFile, ValueError) as exc:
                failures.append(f"{base}: {exc}")
        else:
            raise RuntimeError("Failed to download fastfetch: " + "; ".join(failures))
        target.parent.mkdir(parents=True, exist_ok=True)
        for package in (self.third_party_bin_path.parent, self.third_party_bin_path, target.parent):
            (package / "__init__.py").touch(exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(binary)
            temporary.chmod(0o755)
            os.replace(temporary, target)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        versions.update({"fastfetch": FASTFETCH_VERSION, "fastfetch-platform": platform_name})
        versions_path.write_text(json.dumps(versions, indent=2) + "\n")
        importlib.invalidate_caches()
        return target
