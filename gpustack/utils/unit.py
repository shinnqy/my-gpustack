"""把字节转换成便于阅读的单位"""


BYTES_TO_GIB = 1024 * 1024 * 1024


def byte_to_unit(byte: int, unit: int) -> float:
    return round(byte / unit, 2)


def byte_to_gib(byte: int) -> float:
    return byte_to_unit(byte, BYTES_TO_GIB)
