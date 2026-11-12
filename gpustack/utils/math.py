def largest_power_of_2_leq(n: int) -> int:
    """计算不超过指定数值的最大 2 的幂"""

    if n < 1:
        return 0

    return 1 << (n.bit_length() - 1)
