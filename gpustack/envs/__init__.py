import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true" # ​​DB_ECHO=true​ 会把每条 SQL 打到日志里，调试时很有用
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))

# Server event bus queue capacity. Configurable via env so large clusters can tune the buffer.
"""队列大小"""
EVENT_BUS_SUBSCRIBER_QUEUE_SIZE = int(
    os.getenv("GPUSTACK_EVENT_BUS_SUBSCRIBER_QUEUE_SIZE", 1024)
)

"""用来决定 GPUStack 自动创建的默认集群使用哪种后端类型"""
DEFAULT_CLUSTER_KUBERNETES = (
    os.getenv("GPUSTACK_DEFAULT_CLUSTER_KUBERNETES", "false").lower() == "true"
)