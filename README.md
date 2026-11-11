# My GPUStack：第七章完成版

按 `guide/v1/01`～`07` 的正文及配套辅助代码整理。
已具备集群、模型与实例记录、事件控制器，以及 Worker 注册、设备采集、心跳和状态批量上报。
新增模型和实例的 Watch 流、Worker 客户端缓存、ServeManager 订阅骨架、Server 快照缓存及写入后的缓存失效。
自动创建的模型实例在本阶段仍保持 `pending`；调度、下载和推理进程属于后续章节。

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

## 第七章检查点

重启 Server 和 Worker，使它们加载第七章代码。Worker 加上 `--debug` 可查看缓存和重连日志。

在一个终端打开实例 Watch：

```bash
curl -N 'http://127.0.0.1:30080/v2/model-instances?watch=true'
```

在另一个终端创建模型（需要已创建默认集群）：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"watch-demo","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-0.6B","replicas":1}'
```

Watch 连接会先回放现有实例，再推送新事件：`type: 1` 创建、`type: 2` 更新、`type: 3` 删除。
没有事件时每 15 秒发送空行。模型列表也支持 `/v2/models?watch=true`。

Worker 日志应出现 `Watching model instances event.`、`model-instances cache watch started`。
实例尚未分配给 Worker，所以正常情况下不会出现 `assigned to me`。
重启 Server 后，Worker 会记录连接错误、间隔 5 秒重试，并通过新快照重建缓存。

至少存在一个实例时，可运行教材的 50 个并发 Watch 检查：

```bash
uv run python tests/test_watch_storm.py --server-url http://127.0.0.1:30080
```

预期输出 `50 streams got the snapshot`。Server 的 `--debug` 日志中，一个冷快照应只加载一次。

## 本地验证

```bash
uv run --python 3.12 --isolated pytest -q
```

测试使用临时 SQLite 数据库、真实 HTTP Watch 连接和模拟的 GPU/NPU 探测结果，
覆盖第六章功能，以及事件回放、连接关闭、客户端缓存、过滤、删除和重命名后的缓存失效、
级联删除、并发读写与 50 个 Watch 共用一次快照查询。
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
Worker 停止后，本章不会自动重算离线状态；定期失联检查在第十二章实现。
