import enum


class SourceTypeEnum(str, enum.Enum):
    """
    统一配置来源标签
    """

    BUILTIN = "builtin"
    OFFICIAL = "official"
    FILE = "file"
    URL = "url"
