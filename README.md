# My GPUStack：第六章完成版

按 `guide/v1/01`～`06` 的正文及第六章配套辅助代码整理。
已具备集群、模型与实例记录、事件控制器，以及 Worker 注册、设备采集、心跳和状态批量上报。
模型实例在本阶段仍保持 `pending`；调度、下载和推理进程属于后续章节。

## 安装与启动

需要 Python 3.10～3.12，以及可连接的 PostgreSQL 数据库。

```bash
uv sync
uv run gpustack start \
  --data-dir ./data/server \
  --database-url postgresql://user:password@127.0.0.1:5432/gpustack \
  --token bootstrap
```

首次启动带 `--token` 才会创建默认集群。实际生成的注册令牌保存在 `data/server/token`；
也可以通过 `/v2/clusters` 创建其他集群并读取该集群的注册令牌。

在 GPU/NPU 机器安装项目和驱动后，用对应集群的注册令牌启动独立 Worker：

```bash
uv run gpustack start \
  --data-dir ./data/worker \
  --server-url http://SERVER_IP:30080 \
  --token REGISTRATION_TOKEN
```

Worker 每 30 秒发送心跳和设备状态，Server 每 5 秒批量写入数据库。
重启 Worker 时使用相同的数据目录和注册令牌，保留原 Worker 身份并换发专属令牌。
集群下发的配置会应用到 Worker，本机明确设置的配置优先。

系统信息采集还需要按教材将 fastfetch 可执行文件放到
`gpustack/third_party/bin/fastfetch/fastfetch`（Windows 使用 `fastfetch.exe`）。
系统采集配置已包含在项目中；未安装该工具时仍可注册和上报 GPU，但缺少 CPU、内存等系统信息。

## 本地验证

```bash
uv run --python 3.12 --isolated pytest -q
```

测试使用临时 SQLite 数据库和模拟的 GPU/NPU 探测结果，覆盖注册、重注册、令牌校验、
事件通知、配置优先级、心跳、状态上报、探测失败恢复、集群计数和设备信息转换。
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
Worker 停止后，本章不会自动重算离线状态；定期失联检查在第十二章实现。
