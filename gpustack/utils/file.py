import glob
import os
import re
import shutil


def get_local_file_size_in_byte(file_path):
    if os.path.islink(file_path):
        file_path = os.path.realpath(file_path)

    size = os.path.getsize(file_path)
    return size


def delete_path(path: str):
    """
    Delete a file or directory. If the path is a symbolic link, it will delete the target path.
    """
    if not os.path.lexists(path):
        return

    if os.path.islink(path):
        target_path = os.path.realpath(path)
        os.unlink(path)
        if os.path.lexists(target_path):
            delete_path(target_path)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        for item in os.scandir(path):
            delete_path(item.path)
        shutil.rmtree(path)


def getsize(path: str) -> int:
    """
    Get the total size of the path in bytes. Handles symbolic links and directories.
    """
    # Cache the size of directories to avoid redundant calculations.
    dir_size_cache = {}
    # Keep track of visited directories to avoid infinite loops.
    visited_dirs = set()
    return _getsize(path, visited_dirs, dir_size_cache)


def _getsize(path: str, visited: set, cache: dict) -> int:
    real_path = os.path.realpath(path)

    if os.path.islink(path):
        return _getsize(real_path, visited, cache)
    elif os.path.isfile(real_path):
        return os.path.getsize(real_path)
    elif os.path.isdir(real_path):
        if real_path in visited:
            return 0
        visited.add(real_path)

        if real_path in cache:
            return cache[real_path]

        total = 0
        with os.scandir(real_path) as entries:
            for entry in entries:
                try:
                    total += _getsize(entry.path, visited, cache)
                except FileNotFoundError:
                    pass

        cache[real_path] = total
        return total

    raise FileNotFoundError(f"Path does not exist: {path}")


def get_sharded_file_paths(file_path: str) -> str:
    dir_name, base_name = os.path.split(file_path)
    match = re.match(r"(.*?)-\d{5}-of-\d{5}\.(\w+)", base_name)
    if not match:
        return [file_path]
    prefix = match.group(1)
    extension = match.group(2)
    pattern = os.path.join(dir_name, f"{prefix}-*-of-*.{extension}")
    return sorted(glob.glob(pattern))
