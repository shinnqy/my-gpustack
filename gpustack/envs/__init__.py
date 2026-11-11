import os


"""Configurable environment variables for GPUStack."""


DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"


DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))


DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))


DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))


TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))


EVENT_BUS_SUBSCRIBER_QUEUE_SIZE = int(
    os.getenv("GPUSTACK_EVENT_BUS_SUBSCRIBER_QUEUE_SIZE", 1024)
)


WORKER_HEARTBEAT_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_INTERVAL", 30)
)  # in seconds


WORKER_STATUS_SYNC_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_STATUS_SYNC_INTERVAL", 30)
)  # in seconds


WORKER_HEARTBEAT_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD", 150)
)  # 2.5 minutes in seconds


MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 300)
)  # 5 minutes in seconds


DEFAULT_CLUSTER_KUBERNETES = (
    os.getenv("GPUSTACK_DEFAULT_CLUSTER_KUBERNETES", "false").lower() == "true"
)
