# My GPUStack：第八章完成版

按 `guide/v1/01`～`08` 的正文及配套辅助代码整理。
已具备集群、模型与实例记录、事件控制器，以及 Worker 注册、设备采集、心跳和状态批量上报。
已接通模型和实例的 Watch 流、Worker 客户端缓存、ServeManager 订阅骨架，以及 Server 快照缓存和写入后的缓存失效。
新增调度器、去重队列、集群/GPU 所在 Worker/标签/就绪状态过滤，以及调度结果回写。
有匹配的设备时，实例从 `pending` 经 `analyzing` 推进到 `scheduled`，Worker 收到分配事件并记录日志。
本章候选选择器暂不检查显存容量；资源匹配、下载和推理进程属于后续章节。

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
在第八章，满足调度条件的实例会分配给 Worker，并出现 `assigned to me` 日志。
重启 Server 后，Worker 会记录连接错误、间隔 5 秒重试，并通过新快照重建缓存。

至少存在一个实例时，可运行教材的 50 个并发 Watch 检查：

```bash
uv run python tests/test_watch_storm.py --server-url http://127.0.0.1:30080
```

预期输出 `50 streams got the snapshot`。Server 的 `--debug` 日志中，一个冷快照应只加载一次。

## 第八章检查点

先确认目标集群至少有一台 `ready` Worker，且 `status.gpu_devices` 非空。
使用默认集群的 Worker 时，可以直接创建模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"scheduler-demo","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-0.6B","replicas":1}'
```

如果 Worker 加入的是其他集群，在请求中明确设置对应的 `cluster_id`。
稍等片刻后查询分配结果：

```bash
curl -s http://127.0.0.1:30080/v2/model-instances | python -c '
import json, sys
for item in json.load(sys.stdin)["items"]:
    print(item["name"], item["state"], item["worker_name"],
          item["gpu_indexes"], repr(item["state_message"]))'
```

成功时应为 `scheduled`，包含 Worker 名称和卡号，`state_message` 为空。
Worker 的 ServeManager 日志会记录该实例已分配给自己。

调度器有三条入口：启动时扫描待调度实例、订阅新的 `CREATED` 事件、每 180 秒扫描重试。
评估后通过一个消费者串行放置，同一个实例的不同快照按实例 ID 去重。
如果队列中的实例已经进入其他状态，旧任务不会覆盖它。

给新模型增加一个 Worker 上不存在的 `worker_selector` 标签，可以验证失败提示。
实例会保持 `pending`，`state_message` 说明集群、标签、就绪状态或设备不匹配的原因。
按教材更新 Worker 标签后，下一次符合重试条件的扫描会重新调度。

本章每张可见设备生成一个候选，候选都为 0 分，同分取第一个。
`gpu_selector` 在本章只筛选设备所在的 Worker；具体卡号约束和显存匹配在第九章完善。
Server 已写入的资源声明中显存暂为 0，不能用本章结果判断模型是否实际放得下。

## 本地验证

```bash
uv run --python 3.12 --isolated pytest -q
```

测试使用临时 SQLite 数据库、真实 HTTP Watch 连接和模拟的 GPU/NPU 探测结果，
覆盖第六章功能，以及事件回放、连接关闭、客户端缓存、过滤、删除和重命名后的缓存失效、
级联删除、并发读写与 50 个 Watch 共用一次快照查询。
调度测试还覆盖实例 ID 去重、串行放置、集群隔离、标签和状态过滤、超时重试、
旧任务保护、失败后解除旧绑定，以及调度器停止时清理消费者和定时任务。
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
Worker 停止后，本章不会自动重算离线状态；定期失联检查在第十二章实现。
