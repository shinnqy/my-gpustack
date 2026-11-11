from datetime import datetime
from datetime import timezone
from typing import Optional
from typing import Tuple
import psutil
import socket
from typing import List
import contextlib
import ipaddress


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
