import os
import tempfile

from gpustack.config.config import Config


def test_role_and_dirs():
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(data_dir=d)
        assert cfg.server_role() == Config.ServerRole.SERVER  # 全新目录、没配 server_url
        assert cfg.cache_dir == os.path.join(d, "cache")
        assert cfg.get_api_port() == 30080

        worker = Config(data_dir=d, server_url="http://1.2.3.4")
        assert worker.server_role() == Config.ServerRole.WORKER
        assert worker.get_api_port() == 10150  # worker 监听 worker_port

        both = Config(data_dir=d, enable_worker=True)
        assert both.server_role() == Config.ServerRole.BOTH
        assert both.get_server_url() == "http://127.0.0.1:30080"


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("GPUSTACK_DATABASE_URL", "postgresql://u@h/db")
    with tempfile.TemporaryDirectory() as d:
        assert Config(data_dir=d).get_database_url() == "postgresql://u@h/db"