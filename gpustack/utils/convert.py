def safe_int(value, default=0):
    return safe_convert(value, int, default=default)


def safe_convert(value, target_type, default):
    """
    Safely converts a value to the specified target type.
    If conversion fails, returns the default value.
    """
    try:
        return target_type(value)
    except Exception:
        return default
