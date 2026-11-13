from datetime import datetime
from datetime import timezone
from typing import Optional
from typing import Tuple
import psutil
import socket
from typing import List
import contextlib
import ipaddress
import random
import requests


def get_first_non_loopback_ip(expected_ifname: Optional[str] = None) -> str:
    """
    Get the first non-loopback IPv4 address of the machine.

    Returns:
        The IPv4 address as a string.
    """

    # Fallback to scanning all interfaces
    for name, addrs in psutil.net_if_addrs().items():
        if expected_ifname is not None and name != expected_ifname:
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith(
                ("127.", "169.254.")
            ):
                return addr.address
    if expected_ifname is not None:
        raise Exception(
            f"No non-loopback IPv4 address found on interface {expected_ifname}."
        )
    raise Exception("No non-loopback IPv4 address found.")


def _get_ifname_by_local_ip(
    ip_address: str,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> Optional[str]:
    """
    Given an IP address, return the interface name if it exists and is not loopback/link-local.

    Returns:
        The interface name as a string, or None if not found.
    """

    try:
        ip = ipaddress.ip_address(ip_address)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_link_local:
        return None

    for ifname, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == address_family and addr.address == ip_address:
                return ifname

    return None


def get_ifname_by_ip_hostname(
    ip_address_hostname: str,
    address_family: socket.AddressFamily = socket.AF_INET,
) -> Optional[str]:
    """
    Get the interface name by IP address using psutil.

    Args:
        ip_address_hostname:
            The IP address or hostname to look for. If a hostname is provided, it will be resolved to an IP address.
        address_family:
            The address family (default is socket.AF_INET).

    Returns:
        The interface name associated with the given IP address or hostname.
    """

    local_ifname = _get_ifname_by_local_ip(
        ip_address_hostname, address_family=address_family
    )
    if local_ifname is not None:
        return local_ifname

    cases: List[Tuple[socket.AddressFamily, str]] = [
        (address_family, ip_address_hostname),
    ]
    if address_family == socket.AF_INET:
        cases.append((socket.AF_INET, "8.8.8.8"))
    if address_family == socket.AF_INET6:
        cases.append((socket.AF_INET6, "2001:4860:4860::8888"))

    for af, test_ip in cases:
        with contextlib.suppress(Exception):
            with socket.socket(af, socket.SOCK_DGRAM) as s:
                # the port is arbitrary since we won't actually send any data
                s.connect((test_ip, 1))
                local_ifname = _get_ifname_by_local_ip(s.getsockname()[0], af)
                if local_ifname is not None:
                    return local_ifname

    return None


def parse_port_range(port_range: str) -> Tuple[int, int]:
    """
    解析 40000-40063
    """

    start, end = port_range.split("-")
    return int(start), int(end)


def get_free_port(
    port_range: str,
    unavailable_ports: Optional[set[int]] = None,
    host: str = "127.0.0.1",
) -> int:
    """在范围内随机选择端口，排除内部已占用集合，再通过 TCP 连接探测是否已有服务监听"""

    start, end = parse_port_range(port_range)
    if unavailable_ports is None:
        unavailable_ports = set()

    if len(unavailable_ports) >= end - start + 1:
        raise Exception("No free port available in the port range.")

    while True:
        port = random.randint(start, end)
        if port in unavailable_ports:
            continue
        if is_port_available(port, host):
            return port
        else:
            unavailable_ports.add(port)
            if len(unavailable_ports) == end - start + 1:
                raise Exception("No free port available in the port range.")
            continue


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Test if a port is available.

    Returns:
        True if the port is available, False otherwise.
    """

    # Then, try to connect (if someone is listening, connect will succeed)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.5)
            result = s.connect_ex((host, port))
            if result == 0:
                # Someone is listening, port is not available
                return False
        except Exception:
            pass

    return True


def is_offline(
    last_update: Optional[datetime],
    timeout_seconds: int,
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if the last_update time is offline based on the timeout_seconds.

    Args:
        last_update: The last update time (UTC datetime). If None, it means no record.
        timeout_seconds: The threshold in seconds to consider offline.
        now: The current time (UTC datetime), defaults to datetime.now(timezone.utc)

    Returns:
        Tuple[bool, Optional[str]]: (Whether offline, last_update readable string)
            - If last_update is None, returns "unknown"
            - Otherwise returns formatted time "%Y-%m-%d %H:%M:%S UTC"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if last_update is None:
        return True, "unknown"

    last_update_ts = int(last_update.timestamp())
    now_ts = int(now.timestamp())

    is_offline_flag = (now_ts - last_update_ts) > timeout_seconds
    last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
    return is_offline_flag, last_update_str


def check_registry_reachable(address: str) -> bool:
    """
    请求仓库的 /v2/，把状态码小于 500 视为可达，包括需要认证的 401 响应。它供默认仓库选择逻辑使用
    """
    url = f"{address}/v2/"
    try:
        resp = requests.get(url, timeout=3)
        reachable = resp.status_code < 500
    except Exception:
        reachable = False
    return reachable
