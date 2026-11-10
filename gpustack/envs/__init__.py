import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true" # ​​DB_ECHO=true​ 会把每条 SQL 打到日志里，调试时很有用
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))