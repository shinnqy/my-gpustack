import re

"""从日志文件名解析重启编号，供清理旧日志使用"""

def extract_restart_count(filename: str) -> int:
    """Extract restart count from filename like '123.5.log'.

    Args:
        filename: Log filename in format {id}.{restart_count}.log

    Returns:
        Restart count as integer, or 0 if pattern doesn't match
    """
    match = re.match(r'\d+\.(\d+)\.log', filename)
    return int(match.group(1)) if match else 0


def extract_container_restart_count(filename: str) -> int:
    """Extract restart count from container log filename.

    Args:
        filename: Log filename in format {id}.container.{restart_count}.log

    Returns:
        Restart count as integer, or 0 if pattern doesn't match
    """
    match = re.match(r'\d+\.container\.(\d+)\.log', filename)
    return int(match.group(1)) if match else 0
