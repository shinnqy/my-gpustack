# My GPUStack：第九章完成版

按 `guide/v1/01`～`09` 的正文及配套辅助代码整理。
已具备集群、模型与实例记录、事件控制器，以及 Worker 注册、设备采集、心跳和状态批量上报。
已接通模型和实例的 Watch 流、Worker 客户端缓存、ServeManager 订阅骨架，以及 Server 快照缓存和写入后的缓存失效。
已接通调度器、去重队列、集群/GPU 所在 Worker/标签/就绪状态过滤，以及调度结果回写。
有匹配的设备时，实例从 `pending` 经 `analyzing` 推进到 `scheduled`，Worker 收到分配事件并记录日志。
新增模型配置与分类识别、权重大小估算、显存分配账本、vLLM 单卡/单机多卡匹配、
Spread/Binpack 放置评分，以及缩容评分链。模型文件下载和推理进程在后续章节接入。

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

第八章的零显存候选选择器已由第九章的资源匹配选择器替换。
现在会约束显式选择的卡号，并写入实际的显存预留预算。

## 第九章检查点

Server 需要能够读取模型的 `config.json` 和仓库文件大小。支持 Hugging Face、ModelScope，
以及 Server 能访问的本地模型目录；当前阶段尚未接通从 Worker 远端读取本地模型文件。
本地目录会统计权重文件大小，配置读取和资源估算不会加载模型权重。

LLM 的默认估算为“权重大小 × 1.2 + 2 GiB”。
账本按 `computed_resource_claim` 统计已分配资源，扣除系统预留，计算剩余容量。
vLLM 默认每张选中卡预留总显存的 90%，而且每张卡必须有足够的可分配比例。
Embedding 等非生成模型通常按估算值预留；部分架构保留 GMU 行为。

下面以一台空闲的 **2 × 24 GiB** Worker 为例。实际仓库的权重大小可能与教材的近似值不同，
为稳定复现单卡、多卡和排队行为，示例通过环境变量显式指定显存需求（单位为字节）。
首次创建小模型后，确认它进入 `scheduled` 再创建大模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"fit-small","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-0.6B","env":{"GPUSTACK_MODEL_VRAM_CLAIM":"3221225472"}}'

curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"fit-large","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-8B","env":{"GPUSTACK_MODEL_VRAM_CLAIM":"23300197580"}}'
```

小模型应占一张卡，资源声明约为 `{"0":23192823398}`，分类为 `llm`。
大模型需要约 21.7 GiB，超过单卡 `24 × 0.9 = 21.6 GiB` 的预算；小模型占卡时，
大模型保持 `pending`，错误说明包含估算需求和可分配容量。
删除小模型对应的 **Model** 后，定时重试会将大模型分配到两张卡，每张预留约 21.6 GiB。
模型创建响应中的 `id` 可用于删除：`DELETE /v2/models/{id}`。

再删除大模型，释放两张卡后，用 Binpack 验证两个副本共享同一张卡的调度预算：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"fit-shared","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-0.6B","placement_strategy":"binpack","backend_parameters":["--gpu-memory-utilization=0.4"],"env":{"GPUSTACK_MODEL_VRAM_CLAIM":"3221225472"},"replicas":2}'
```

两个副本可以分配到同一张卡，各预留 `24 GiB × 0.4`。
默认 Spread 优先把同一模型的副本分散到不同 Worker；Binpack 优先利用已有占用的卡。
多卡选择同时检查注意力头数、词表等 TP 整除约束，并优先保留用卡最少的可行方案。
显式设置卡数或 TP 时，选择器会遵守要求，不会增加卡数来绕过整除失败。

`computed_resource_claim.vram` 是调度预算。模型权重大小仍是经验估算，
获取仓库大小失败时会退回框架开销；需要确定的预算时，可使用 `GPUSTACK_MODEL_VRAM_CLAIM`。
当前 Worker 收到分配后仍只记录日志，不会启动推理容器。

## 本地验证

```bash
uv run --python 3.12 --isolated pytest -q
```

测试使用临时 SQLite 数据库、真实 HTTP Watch 连接和模拟的 GPU/NPU 探测结果，
覆盖第六章功能，以及事件回放、连接关闭、客户端缓存、过滤、删除和重命名后的缓存失效、
级联删除、并发读写与 50 个 Watch 共用一次快照查询。
调度测试还覆盖实例 ID 去重、串行放置、集群隔离、标签和状态过滤、超时重试、
旧任务保护、失败后解除旧绑定，以及调度器停止时清理消费者和定时任务。
资源匹配测试覆盖显存预留与扣减、统一内存、异构卡分组、GMU 边界、单卡/多卡、
手动选卡、TP 整除、模型分类、重调度释放自身预留、放置评分和缩容顺序。
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
Worker 停止后，本章不会自动重算离线状态；定期失联检查在第十二章实现。
