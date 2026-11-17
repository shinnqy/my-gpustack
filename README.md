# My GPUStack：第十五章完成版

按 `guide/v1/01`～`15` 的正文及配套辅助代码整理。
已具备集群、模型与实例记录、事件控制器，以及 Worker 注册、设备采集、心跳和状态批量上报。
已接通模型和实例的 Watch 流、Worker 客户端缓存、ServeManager 订阅骨架，以及 Server 快照缓存和写入后的缓存失效。
已接通调度器、去重队列、集群/GPU 所在 Worker/标签/就绪状态过滤，以及调度结果回写。
有匹配的设备时，实例从 `pending` 经 `analyzing` 推进到 `scheduled`，Worker 接手后继续准备模型文件。
新增模型配置与分类识别、权重大小估算、显存分配账本、vLLM 单卡/单机多卡匹配、
Spread/Binpack 放置评分，以及缩容评分链。
新增 ModelFile 记录与关联、文件控制器、进程池下载、进度与错误同步、同机文件复用和缓存清理。
已接通独立启动子进程、等待文件就绪、端口预留、vLLM 镜像和参数解析，以及 Runtime 工作负载创建。
已接通推理就绪检查、运行状态同步、错误重启与指数退避、删除后的容器收割，
以及 Worker 失联检测、维护模式和离线实例清理。
文件准备完成后，启动子进程创建推理容器并结束初始化；容器运行且就绪检查通过后，实例进入 `running`。
已接通按模型名选择实例的 OpenAI 兼容接口、轮询负载均衡、Server 到 Worker 再到推理引擎的两跳代理，
支持普通 JSON、SSE 流式响应、二进制响应和表单文件上传。
已接通多机候选选择、从属 GPU 资源记账、各 Worker 的文件就绪同步、主从启动与状态协调，
以及 vLLM 的 Ray 和 MP 多机工作负载计划。
已接通 Kubernetes 集群选项和部署清单、经 Worker 的 Kubernetes API 代理，
以及 Server 侧 operator worker gateway 子进程、TLS Unix socket 客户端和集群订阅协调。

## 安装与启动

以下是源码开发方式；使用第十五章的完整运行环境时，可按后文的
[统一镜像启动 Server](#统一镜像启动-server) 部署。

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

系统信息采集需要 fastfetch，可沿用原项目的工具准备命令：

```bash
uv run gpustack download-tools
```

命令按当前操作系统和架构下载到 `gpustack/third_party/bin/fastfetch/fastfetch`
（Windows 为 `fastfetch.exe`）。也支持原项目的 `--system`、`--arch`
和 `--tools-download-base-url` 参数；当前阶段只准备已使用的 fastfetch。
未准备工具时仍可注册和上报 GPU，但缺少 CPU、内存等系统信息。

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
在第十章，Worker 会立即接手准备文件，查询时可能已经进入 `initializing`、`downloading` 或 `starting`。

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
当前 Worker 收到分配后会准备文件并创建推理容器。
删除 Model 后，周期性状态同步会停止对应容器，释放实际显存。

## 第十章检查点

先执行 `uv sync` 并重启 Server 和 Worker。
Server 启动时会为第九章的旧实例表补齐四个可空字段，并创建模型文件与两张关联表，保留已有模型和实例数据。
新建的 `source_index` 是普通索引，同一来源可以在不同 Worker 上有独立文件记录。

Worker 需要能够访问下载源，并能写入自己的缓存目录。默认下载位置为：

- Hugging Face：`<worker-cache-dir>/huggingface/<组织>/<仓库>`。
- ModelScope：`<worker-cache-dir>/model_scope/<组织>/<仓库>`。
- 本地路径：直接使用 Worker 上已有的路径，不复制文件；当前资源匹配还要求 Server 能读取模型配置。

创建一个资源能放下的小模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"files-demo","source":"huggingface","huggingface_repo_id":"Qwen/Qwen3-0.6B","backend_parameters":["--gpu-memory-utilization=0.4"]}'
```

在另一个终端观察实例事件：

```bash
curl -N 'http://127.0.0.1:30080/v2/model-instances?watch=true'
```

另外查询文件记录：

```bash
curl -s 'http://127.0.0.1:30080/v2/model-files' | python -m json.tool
```

完整状态链为 `scheduled → initializing → downloading → starting`。
文件就绪后，实例的 `download_progress` 为 100，`resolved_path` 指向 Worker 上的实际模型路径。
下载文件的记录包含 `size`、`resolved_paths` 和 `state=ready`；进度在下载收尾前最多显示 99%。

同一 Worker 上的同一来源只需要一份 ModelFile，多个实例通过关联表复用它。
将模型副本数改为 2 时，资源预算允许的情况下，第二个实例复用已有文件；已就绪的文件会直接推进实例到 `starting`。
两台 Worker 则分别维护各自的文件记录。

文件下载出错时，ModelFile 和关联实例都会进入 `error`。
让实例重新经过 `initializing` 会重置出错文件并重新提交下载；自动错误重启属于第十二章。
Worker 的文件 Watch 断线后会重连，停止时会取消下载并关闭下载进程池和跨进程管理器。

删除模型或缩容会解除实例与文件的关联，保留缓存供后续复用。
文件仍被主模型或草稿模型引用时，删除接口返回 409。
确认文件不再使用后，可以按文件 ID 删除记录，并请求 Worker 清理磁盘缓存：

```bash
curl -X DELETE 'http://127.0.0.1:30080/v2/model-files/FILE_ID?cleanup=true'
```

省略 `cleanup=true` 时保留磁盘文件。实例到达 `starting` 表示文件准备完成，
第十一章的启动子进程会继续创建推理工作负载。

## 第十一章检查点

执行 `uv sync` 后重启 Server 和 Worker。
Worker 需要能访问容器运行时；使用 Docker 时，先确认 `docker info` 正常。
NVIDIA 机器需要驱动和 NVIDIA Container Toolkit；昇腾机器需要驱动和 Ascend Docker Runtime。
Server 可以部署在其他机器上。

创建一个资源能放下的模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"engine-demo","source":"model_scope","model_scope_model_id":"Qwen/Qwen3-0.6B"}'
```

Worker 为实例预留服务端口，启动独立子进程，再把实例改为 `initializing`。
子进程通过 Watch 等待模型文件进入 `starting`，构造 Runtime 计划并创建容器，然后结束初始化。
端口默认在 `40000-40063` 内分配；实例记录包含 `port`、`ports` 和启动子进程的 `pid`。

镜像优先级为：模型的 `image_name`、后端版本配置、`gpustack-runner` 内置镜像目录。
目录解析会考虑计算后端、宿主平台、运行时版本和昇腾变体。
可以在模型创建请求中设置完整的 `image_name` 来指定已有镜像。

在 Worker 上按实际实例名查看容器和命令：

```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
docker inspect INSTANCE_NAME --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
docker logs -f INSTANCE_NAME
```

默认命令使用 `vllm serve <模型路径>`，补齐服务地址、端口和模型名；多卡自动设置 TP。
当模型上下文大于 8192 且用户未设置时，默认限制为 8192。
用户的 `backend_parameters` 可覆盖默认地址、端口、上下文等参数。
Runtime 的设备资源键只包含被调度选中的卡，模型目录通过挂载传入容器。

昇腾使用 `huawei.com/devices`，并注入 `TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF` 等变量。
310P 支持完整 SoC 名称和 `310p` 变体标识，自动使用 `--enforce-eager --dtype float16`。
NVIDIA 使用 `nvidia.com/devices`。

等 vLLM 日志显示服务启动完成后，直接请求 Worker 上分配的端口：

```bash
curl http://WORKER_IP:INSTANCE_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"engine-demo","messages":[{"role":"user","content":"你好"}]}'
```

启动日志在 `<worker-data-dir>/log/serve/<实例ID>.<重启次数>.log`，例如 `1.0.log`。
第十二章已接通自动置为 `running`、错误重启和删除后的容器清理，检查方法见下文。
GPUStack 入口 `/v1/chat/completions` 的代理已接通，检查方法见第十三章。

## 第十二章检查点

执行 `uv sync` 后重启 Server 和 Worker，启动方式见上文。启动时会为旧 Worker 表补齐
`maintenance`、`proxy_mode` 和 `proxy_address` 三个可空字段，保留已有记录。

按第十一章示例创建模型后，观察实例和模型：

```bash
curl -s http://127.0.0.1:30080/v2/model-instances | python -m json.tool
curl -s http://127.0.0.1:30080/v2/models | python -m json.tool
```

Worker 默认每 3 秒检查工作负载。vLLM 容器运行且 Worker 地址上的 `/v1/models` 返回 200 后，
实例从 `starting` 变为 `running`，模型的 `ready_replicas` 随之增加。
容器失败、退出或消失时，实例进入 `error`，`state_message` 保留运行时原因和退出码。

用无效参数验证错误恢复：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"bad","source":"model_scope","model_scope_model_id":"Qwen/Qwen3-0.6B","backend_parameters":["--no-such-flag"]}'
```

默认 `restart_on_error=true`。Worker 每 10 秒处理错误实例，经 `scheduled` 重新走启动流程；
首次无需退避，连续失败后依次等待 10、20、40 秒，最长 300 秒。
观察 `restart_count`、`last_restart_time` 和编号日志，确认在重试；
设置模型的 `restart_on_error=false` 可停止自动重启。实例成功进入 `running` 后清空连续失败退避。

删除模型时，Worker 周期性收割 Server 已不再分配给自己的实例并停止容器。
清理前会用不分页、绕过客户端缓存的请求确认删除；Server 请求失败不会当作“所有实例已删除”。
Worker 还每 120 秒检查遗留工作负载，启动后延迟 15 秒，默认只清理创建超过 300 秒的孤儿。

在 Worker 终端按 Ctrl-C，观察 `/v2/workers` 和实例状态：

- Server 每 15 秒探测 Worker 的 `/healthz`；不可达时将 Worker 标为 `unreachable`。
- 心跳中断超过默认 150 秒后，Worker 变为 `not_ready`，其上的实例标为 `unreachable`。
- Server 每 30 秒清理心跳中断超过 300 秒的非维护 Worker 上的实例；模型控制器补齐副本，再由调度器选择可用 Worker。
- 清理前恢复 Worker 时，使用原数据目录、名称和端口启动；心跳和就绪检查恢复后，实例回到 `running`，复用仍存在的推理容器。

主动连通性探测默认 `auto`：Worker 数量超过 50 时跳过主动探测，心跳检查仍运行。
可通过 `GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE=enabled` 或 `disabled` 明确控制主动探测。

计划停机前，可按查询到的 Worker ID 和现有名称启用维护模式：

```bash
curl -s -X PUT http://127.0.0.1:30080/v2/workers/WORKER_ID \
  -H 'Content-Type: application/json' \
  -d '{"name":"GPU_NODE_NAME","maintenance":{"enabled":true,"message":"计划维护"}}'
```

维护 Worker 不会因离线被自动删除实例，也不接收新的调度。
恢复后用同一接口把 `maintenance.enabled` 改为 `false`。未提供的标签等字段保留原值。

可选的主动推理检查默认关闭。在模型 `env` 中设置
`GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED=true` 后，会按模型类别发起小请求，
默认间隔 300 秒、超时 15 秒、连续失败 3 次才将实例标为 `error`。
对应的 `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL`、`_TIMEOUT` 和 `_FAILURE_THRESHOLD`
可分别覆盖这些值。推理检查也走前述错误恢复流程。

## 第十三章检查点

执行 `uv sync` 后，用上文的命令分别启动 Server 和 Worker。
Worker 的固定 API 端口（默认 10150）需能被 Server 访问。
通过 ModelScope 创建一个能放入设备的小模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"proxy-demo","source":"model_scope","model_scope_model_id":"Qwen/Qwen3-0.6B"}'
```

等实例进入 `running` 后，在能访问 Server 的机器上执行：

```bash
curl -s http://127.0.0.1:30080/v1/models
curl http://127.0.0.1:30080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"proxy-demo","messages":[{"role":"user","content":"你好"}]}'
curl -N http://127.0.0.1:30080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"proxy-demo","stream":true,"messages":[{"role":"user","content":"数到十"}]}'
```

`/v1/models` 只列出已有就绪副本的模型。流式请求应逐步返回 `data: ...`，
最后携带 token 用量并以 `data: [DONE]` 结束；引擎需支持相应的响应格式。
表单上传、重复表单字段、二进制响应、引擎错误状态码和多值响应头均可经过代理。
流式连接断开时关闭上游连接；请求转发开始前释放数据库会话。

也可运行教材的 OpenAI SDK 检查脚本：

```bash
uv run python tests/test_openai_client.py \
  --base-url http://127.0.0.1:30080/v1 --model proxy-demo
```

本章 Server 按模型名直接查找模型，省略用户/API Key 权限检查，因此 SDK 示例使用 `api_key="unused"`。
Server 会把请求中的用户 Authorization 替换为 Worker 专属令牌，并添加实例路由头；
Worker 验证令牌后，只向本机 `running` 实例转发，记录成功请求供主动健康检查使用。

资源允许时把副本数改为 2，连续发送 4 次请求，应在两个实例间轮询。
实例离开 `running` 后，其查询缓存失效，后续请求使用剩余就绪实例。
不存在的模型返回 OpenAI 格式 404；副本数为 0 或全部实例未就绪时返回 503。
直接请求 Worker 推理接口且不带令牌时返回 401。
上游返回 4xx/5xx 时保留状态和正文；连接失败返回 503，超时返回 504。
流式响应已经开始后发生故障时，通过 SSE 错误事件报告，不再次发送 HTTP 响应头。

`/v1-openai` 保留教材列出的旧路径子集，转给引擎时统一使用 `/v1`。
当前实现的是 `worker` 两跳模式；`direct`、`tunnel`、Higress 网关和专用 `/rerank` 路由不在本章范围。
具体模型是否支持音频、图像等接口，由其推理引擎决定。

## 第十四章检查点

先启动 Server，再在两台 GPU 机器上执行上文的 Worker 启动命令，使用同一集群的注册令牌。
可分别加 `--worker-name gpu-1` 和 `--worker-name gpu-2`，确认 `/v2/workers` 中两台机器均为 `ready`，
且 `status.gpu_devices` 包含实际设备。宿主机不需要单独安装 Ray，推理镜像需要包含所选的执行后端。

机器之间需要互通 Worker API 端口（默认 10150）、服务端口段（默认 40000–40063），
Ray 模式还需要互通 Ray 端口段（默认 41000–41999）。
Ray 模式要求各机器上的模型绝对路径一致：配置相同的缓存目录位置，各自下载自己的模型文件。
MP 模式按各自 Worker 的 ModelFile 记录查找文件，可使用不同缓存目录。

下面按教材示例部署到两台各一张卡的机器上；请按实际显存选择模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"qwen3-32b-2node","source":"model_scope","model_scope_model_id":"Qwen/Qwen3-32B","backend_parameters":["--tensor-parallel-size=1","--pipeline-parallel-size=2","--distributed-executor-backend=ray"]}'
```

`TP × PP = 2` 强制使用两张卡；两台机器都只有一张卡时会选择多机组合。
自动选择依次尝试单卡、单机多卡、多机多卡；多机自动选择要求 GPU 类型和每台参与卡数相同，
每张卡满足显存预算，总预算足够，并符合 TP 整除及 MP 拓扑约束。
`distributed_inference_across_workers=false` 时不选择多机；显式选卡支持跨 Worker。
从属卡的资源声明也计入账本，避免后续实例重复占用。

查询实例和文件记录：

```bash
curl -s http://127.0.0.1:30080/v2/model-instances | python -m json.tool
curl -s http://127.0.0.1:30080/v2/model-files | python -m json.tool
```

应看到一个实例：`worker_id` 指向主机器，`distributed_servers.mode=initialize_later`，
`subordinate_workers` 中有另一台机器的 GPU、资源声明、进度、PID 和状态。
同一个来源在两台 Worker 上各有一条 ModelFile，全部主模型和草稿文件就绪后才进入 `starting`。

Worker 根据自己的角色启动工作负载：

- Ray 主机器运行 vLLM 和 `ray-head` 两个容器，共享 `/tmp` 卷；从属工作负载名以 `-f0` 结尾，执行 `ray start --address=<主机器IP>:41000`。
- MP 模式每台机器运行 vLLM，按拓扑生成 `--nnodes`、`--node-rank`、主地址、主端口以及所需 DP 参数；从属进程使用 `--headless`。
- NVIDIA 配置 NCCL/Gloo 网卡；昇腾配置 HCCL/Gloo 网卡。用户明确指定的通信配置保留。

Ray 主机器上执行 `docker exec RAY_HEAD_CONTAINER ray status`，应能看到两个 Ray 节点。
主实例只有在从属节点状态正常、主容器运行并通过 HTTP 就绪检查后才进入 `running`。
从属节点的 `running` 表示其启动流程已接手；最终推理就绪仍由主机器的引擎检查决定。
随后可以通过第十三章的 `/v1/chat/completions` 请求 `qwen3-32b-2node`。

验证 MP 时，释放前一个模型的资源，再将请求中的 `--distributed-executor-backend=ray` 改为 `mp`，
并选择支持这些多机参数的 vLLM 版本和镜像。支持 TP/PP、纯 DP 和 DP 嵌套 TP/PP 三种拓扑。
程序根据显式参数或 `backend_version` 选择执行后端；本项目当前默认规则中，带 `-custom` 且版本达到 0.18.0 时默认 MP，其余默认 Ray。
MP 端口布局为 HTTP、DP RPC、PyTorch master、VLLM_PORT、连接端口，并为 DP 主端口预留连续十个端口。

停止从属工作负载后，应先看到其状态和错误原因写入 `subordinate_workers`，
主实例随之进入 `error`，随后按第十二章的退避规则整体重启主从工作负载。
短暂停止 Worker 进程会标记从属节点和实例 `unreachable`；恢复 Worker 后可复用仍存在的工作负载。
删除模型后，两台 Worker 都会收割各自的工作负载。

主从状态通过字段补丁更新，Server 合并嵌套字段并校验 `restart_count`，避免并发回写互相覆盖，
也避免旧启动进程修改新一轮重启的数据。
每台 Worker 的 `<data-dir>/log/serve` 保存启动日志和容器日志，Ray head 日志如 `1.container.ray-head.0.log`；
重启时保留当前和上一轮，删除实例时清理这些日志。

## 镜像构建：补齐第十五章的打包链

原 GPUStack 使用同一个 `gpustack/gpustack` 镜像运行 Server 和 Worker，
由 `gpustack start` 的参数及环境变量决定角色。本项目沿用这一结构：

| 原项目入口 | 本项目对应实现 |
|---|---|
| `Makefile → hack/build.sh` | `make build` 构建 wheel 和源码包 |
| `Makefile → hack/package.sh` | `make package` 调用 Buildx 构建并加载镜像 |
| `pack/Dockerfile` 的 `base → gpustack` | 安装本项目 wheel、准备目标架构工具、复制 Operator 二进制 |
| `pack/entrypoint.sh` | 将容器参数交给既有 `gpustack start`，用于两种角色 |

这里迁入第 1～15 章已用到的依赖；原镜像中的 UI、Higress、内置 PostgreSQL 和
S6 多服务编排尚未进入本项目。当前镜像继续使用外部 PostgreSQL。
Docker 构建时不复制宿主机的虚拟环境、模型缓存、数据库或其他架构的 fastfetch。

沿用原项目的构建参数和默认镜像名：

```bash
make build
make package
```

默认构建 `gpustack/gpustack:dev`，并通过 `--load` 加载到当前 Docker 引擎；
`PACKAGE_NAMESPACE`、`PACKAGE_REPOSITORY`、`PACKAGE_TAG` 可修改镜像名，
`PACKAGE_ARCH` 支持 `amd64` 和 `arm64`。只有显式设置
`PACKAGE_PUSH=true` 时才推送仓库。

`GPUSTACK_OPERATOR_VERSION` 是镜像内 Operator 二进制的构建参数，
可覆盖 Dockerfile 的版本；Kubernetes 中 Operator 的镜像仍由已有
`k8s_options.operatorImage` / `GPUSTACK_OPERATOR_IMAGE` 配置控制。
Dockerfile 和 Python 版本常量的默认值统一为本次已验证可获取的 `v0.9.2`。

在本次昇腾 ARM64 实验环境，已缓存可用的 Operator `v0.9.2`。
若它只存在于 containerd，先把缓存导入 Docker，再构建：

```bash
ctr -n k8s.io images export --platform linux/arm64 - \
  docker.io/gpustack/gpustack-operator:v0.9.2 | docker load

PACKAGE_ARCH=arm64 \
PACKAGE_PULL=false \
PACKAGE_NETWORK=host \
APT_MIRROR=https://mirrors.aliyun.com \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
GPUSTACK_OPERATOR_VERSION=v0.9.2 \
GPUSTACK_TOOLS_DOWNLOAD_BASE_URL=https://gpustack-1303613262.cos.ap-guangzhou.myqcloud.com \
make package
```

`PACKAGE_PULL=false` 允许使用本地已有的基础镜像；尚未缓存的其他构建依赖仍需下载。
`PACKAGE_NETWORK=host` 为构建中的 `RUN` 使用宿主机网络，并声明 `network.host` entitlement，
对应原项目的 host 构建网络方式。宿主机联网正常、构建容器 DNS 失败时可使用此选项。
其默认值为 `default`，也支持 `none`；它不修改 Docker daemon、iptables 或部署后 Pod 的网络。
`APT_MIRROR` 是 Debian 镜像站的基础地址，同时用于主仓库和安全更新仓库；默认使用官方源。
`PIP_INDEX_URL` 同时用于安装 uv 的 pip，以及随后构建 wheel、安装项目依赖的 uv，默认使用 PyPI。
软件源只用于构建。系统包安装与 uv 安装拆成独立层，后者下载失败后不必重复安装系统包。
默认 `base` 使用 Python 3.11，`PYTHON_VERSION` 可覆盖其版本；
`GPUSTACK_BASE_IMAGE` 保留原项目的基础镜像覆盖能力，自定义基础镜像需具备对应
Python、uv、make、tini 和运行依赖。

构建时会检查 CLI、fastfetch 的系统信息采集和 Operator 二进制。也可检查最终产物：

```bash
docker run --rm gpustack/gpustack:dev --help
docker run --rm --entrypoint gpustack-operator gpustack/gpustack:dev --version
```

Kubernetes 使用 containerd 时，还要把构建产物导入其 `k8s.io` 命名空间：

```bash
docker save gpustack/gpustack:dev | ctr -n k8s.io images import -
crictl images
```

`gpustack/gpustack:dev` 是开发镜像约定，不表示远端仓库中必然存在这个标签。
本地构建并导入后，清单中的 `imagePullPolicy: IfNotPresent` 可以直接使用它。
多节点需在每个匹配节点导入对应架构镜像，或者显式推送到各节点可访问的镜像仓库。
重用同一个开发标签更新代码后，需重新导入并滚动重启 Worker Pod。

## 统一镜像启动 Server

沿用原项目的 `docker-compose/docker-compose.server.yaml` 入口，Server 和 Worker
使用同一个构建产物。镜像已包含 `/usr/bin/gpustack-operator`，Server 会自动启动
`gpustack-operator wg`；容器部署不需要向宿主机复制 Operator 或修改宿主机 PATH。

此 Compose 示例面向 **Linux（包括当前 Ascend 机器）**，使用 host 网络：Server 可以继续连接
宿主机 `127.0.0.1:5432` 上的 PostgreSQL，并使用宿主机的 30080 端口提供 API。
当前阶段仍使用外部 PostgreSQL，镜像中没有内置数据库、UI 或 Higress。
macOS/Windows 的源码开发继续使用上文的启动方式；此 Linux 示例不依赖 Docker Desktop 的 host 网络支持。

先按上一节构建并加载当前机器架构的镜像，再从仓库根目录准备配置：

```bash
cp -n docker-compose/.env.example docker-compose/.env
```

编辑 `docker-compose/.env`，填写 `GPUSTACK_DATABASE_URL` 和
`GPUSTACK_SERVER_EXTERNAL_URL`。新安装默认使用 `gpustack-data` 数据卷；迁移已有 Server 时，
数据库 URL 和宿主机数据目录都复用原值，以保留集群、Worker 注册信息、令牌和服务端密钥。
如果原源码启动命令没有指定数据库 URL，本阶段的默认值是
`postgresql://root@127.0.0.1:5432/gpustack?sslmode=disable`。

当前 Ascend 环境的 `.env` 可以使用以下网络与目录设置，数据库地址按原启动参数填写：

```dotenv
GPUSTACK_SERVER_DATA_DIR=/data2/qy/scheduling/tmp/gpustack-server
GPUSTACK_SERVER_EXTERNAL_URL=http://100.107.227.30:30080
GPUSTACK_API_PORT=30080
```

默认镜像为 `docker.io/gpustack/gpustack:dev`，与 `make package` 的默认产物相同。
`IMAGE_REGISTRY`、`IMAGE_NAMESPACE`、`GPUSTACK_TAG` 沿用原项目的 Compose 配置方式；
自定义完整镜像名称时设置 `GPUSTACK_IMAGE`。Compose 使用本地镜像；使用仓库镜像时先执行
`docker pull <完整镜像名>`。同一镜像名称也会通过 `GPUSTACK_IMAGE_NAME_OVERRIDE` 用于生成
Worker 清单，集群自身设置的镜像覆盖值仍优先。

若模型使用 `/var/lib/gpustack/models/...` 这样的本地路径，Server 的模型分析器也需要读取权重目录。
在 Compose 的 `volumes` 中启用示例挂载：

```yaml
      - /var/lib/gpustack/models:/var/lib/gpustack/models:ro
```

这样 Server 可读取 Ascend 宿主机上的同一批模型，Worker 仍使用自己的数据目录。
其他本地路径也按相同路径挂载；从模型仓库下载的模型不要求这项额外挂载。

检查配置后，在原终端停止源码 Server，使 API 端口和数据目录交由容器 Server 使用，再启动容器：

```bash
docker compose --env-file docker-compose/.env \
  -f docker-compose/docker-compose.server.yaml config --quiet

docker compose --env-file docker-compose/.env \
  -f docker-compose/docker-compose.server.yaml up -d
```

Compose 明确关闭内嵌 Worker；已有的 Kubernetes Worker DaemonSet 继续连接同一个 Server 地址。
`GPUSTACK_GATEWAY_MODE=disabled` 关闭的是当前阶段尚未打包的 Higress 入口，与自动启动的
Operator worker gateway（`gpustack-operator wg`）是两个不同组件。

验证 API、Operator 二进制和实际 gateway 子进程：

```bash
docker compose --env-file docker-compose/.env \
  -f docker-compose/docker-compose.server.yaml ps
docker logs --tail 100 gpustack-server
curl -fsS http://127.0.0.1:30080/healthz
docker exec gpustack-server gpustack-operator --version
docker top gpustack-server -eo pid,ppid,args
```

预期容器健康、API 返回 200，进程列表中出现 `gpustack-operator wg ... --bind-unix-path ...`。
容器健康检查验证 API；gateway 进程由最后一条命令单独确认。
当前 `k8s-a` 集群还可以检查已有 Worker 和代理：

```bash
curl -fsS http://127.0.0.1:30080/v2/workers
curl -fsS http://127.0.0.1:30080/v2/clusters/2/proxy/api/v1/nodes
```

使用其他集群时替换 ID。仅配置了 `k8s_options.gpuInstanceOptions` 的集群才会订阅 Operator
gateway，因此普通模型服务集群不以 “Subscribed cluster” 日志作为成功条件。

重新构建同一标签后，更新 Server 容器：

```bash
docker compose --env-file docker-compose/.env \
  -f docker-compose/docker-compose.server.yaml up -d --force-recreate
```

## 第十五章检查点

Server 按上一节使用统一镜像和外部 PostgreSQL 启动。准备可用的 Kubernetes 集群、NFD 和对应厂商的驱动/设备运行时。
Server 必须能访问 Worker 节点的 API 端口（默认 10150），Worker Pod 也必须能访问 Server。
当前通过 Worker 固定端口直连，未接入 NAT 隧道。

统一镜像已包含 Server 侧 Operator 网关所需的二进制，当前默认版本为 `v0.9.2`。
保留源码启动方式进行调试时，需要另行在 Server 的 PATH 中准备匹配操作系统和架构的
`gpustack-operator`；找不到时，Server 会按原项目的行为跳过网关子进程和订阅。

创建 Kubernetes 集群，使用 Kubernetes 节点可访问的 Server 地址：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/clusters \
  -H 'Content-Type: application/json' \
  -d '{"name":"k8s-a","provider":"Kubernetes","server_url":"http://SERVER_IP:30080","worker_config":{"image_name_override":"REGISTRY/my-gpustack:chapter15"}}'
```

将 `image_name_override` 替换为上一步构建并分发的镜像。
如果已按默认名称构建并导入 `gpustack/gpustack:dev`，可省略该覆盖值；
若使用自定义仓库或标签，则填入实际镜像名称。
`k8s_options` 支持 `namespace`、`nodeSelector`、`imageCredentials`、`volumeMounts`、`operatorImage` 和 `operator.env`。
省略挂载时默认使用宿主机 `/var/lib/gpustack`；第一项始终挂到容器的 `/var/lib/gpustack`，用于持久化模型缓存。
可修改第一项的宿主机目录，后续挂载可使用本章定义的 hostPath、PVC 或 ConfigMap。

用创建结果中的 ID 替换 `CLUSTER_ID`，下载清单：

```bash
curl -fsS 'http://127.0.0.1:30080/v2/clusters/CLUSTER_ID/manifests?runtime=nvidia' > gpustack-k8s.yaml
kubectl apply -f gpustack-k8s.yaml
kubectl -n gpustack-system get ds,job,deploy,pods -o wide
```

昇腾使用 `runtime=ascend`，多厂商可重复传参，如 `runtime=nvidia&runtime=ascend`。
CPU DaemonSet 始终渲染，只选择 `feature.gpustack.ai/acceleratable=false` 的节点；
GPU DaemonSet 按厂商 PCI 标签选择节点，并通过跨 DaemonSet 的反亲和配置避免同节点重复运行 Worker。
修改命名空间时，后续 kubectl 命令也使用对应的命名空间。

清单包含 Namespace、注册令牌和镜像仓库 Secret、Worker Service/ServiceAccount/RBAC、
各厂商 Worker DaemonSet，以及 operator 的配置和安装 Job。
安装 Job 在集群内使用 ytt/kubectl 创建 operator worker Deployment 等资源。
Worker 的 ServiceAccount 绑定 `cluster-admin`，用于本章的集群 API 代理。

Worker Pod 会以节点名注册，清单设置 `GPUSTACK_RUNTIME_DEPLOY=Kubernetes` 和镜像部署模式。
在 `/v2/workers` 确认对应 `cluster_id` 的 Worker 为 `ready` 后，指定该集群创建模型：

```bash
curl -s -X POST http://127.0.0.1:30080/v2/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"k8s-qwen","cluster_id":CLUSTER_ID,"source":"model_scope","model_scope_model_id":"Qwen/Qwen3-0.6B"}'
kubectl -n gpustack-system get pods -o wide
```

Worker 仍创建通用 Runtime 工作负载计划，由 Kubernetes 部署器生成推理 Pod，
继承 Worker Pod 的数据卷、环境配置和镜像拉取凭证，并固定到相应节点。
当前 Runtime 在 operator 设备插件模式下会请求如 `nvidia.com/gpu.shared` 的资源并附带首选设备索引；
具体资源键和注入方式取决于节点插件及 Runtime 策略，不能一律按原生 `nvidia.com/gpu` 判断。
实例进入 `running` 后，使用第十三章的 `/v1/chat/completions` 请求模型。

集群代理示例：

```bash
curl -s 'http://127.0.0.1:30080/v2/clusters/CLUSTER_ID/proxy/api/v1/nodes'
curl -N 'http://127.0.0.1:30080/v2/clusters/CLUSTER_ID/proxy/api/v1/pods?watch=true'
```

Server 从该集群选择 ready Worker，使用其专属令牌访问 `/cluster-proxy/...`；
Worker 验证请求后，使用 Pod 的 ServiceAccount 令牌和 CA 访问 API Server。
令牌每次请求重新读取，支持 kubelet 轮换；watch/日志请求原样流式返回，不占用数据库会话，断开时关闭上游。
本章 Server 管理接口仍省略用户/多租户权限检查，GPU 实例 CRUD 和 vGPU 调度入口属于后续章节。

`k8s_options.gpuInstanceOptions` 存在（可为 `{}`）且至少有一台 ready Worker 时，
Server 会把集群订阅给本机的 operator 网关；模型服务集群未设置该字段时不订阅。
重复状态事件不会重复订阅，失去最后一台 ready Worker 时取消；失败调用由心跳重试。
Server 退出时停止网关并清理私有 socket 目录。

启动会给已有 `clusters` 表补上可空的 `k8s_options`，保留原集群、注册令牌和已有模型。
安装包已包含四个 jinja 模板，安装后也能生成清单。

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
模型文件测试覆盖共享和多 Worker 记录、提交关联后发布事件、进度汇总、草稿文件、
下载失败与重试、缓存清理、下载取消、文件锁、进程池资源关闭，以及保留数据的旧库升级。
进程验证使用小型 HTTP 测试仓库、真实进程池和模拟 NPU，验证下载字节、文件复用及 `starting` 状态。
启动测试覆盖 CUDA/CANN 设备选择、挂载和端口、vLLM 参数覆盖、310P 参数与环境变量、
镜像优先级、运行时版本匹配、子进程启动顺序、端口复用及失败后的状态回写。
第十一章使用模拟 Runtime 接口验证工作负载计划，不拉取推理镜像；真实 GPU/NPU 推理需要在对应机器上验收。
健康恢复测试覆盖运行时状态与 HTTP 就绪、错误原因、指数退避、恢复保护、缓存重连时的收割确认、
可选主动推理检查、心跳恢复、维护接口、集群隔离、离线实例重建、孤儿清理及保留数据的 Worker 表升级。
第十二章独立 Server/Worker 进程检查使用临时 SQLite、模拟 Runtime 和本地 HTTP 推理替身，
验证就绪、故障重启、Worker 失联与重连、复用原工作负载，以及删除模型后的进程清理。
推理代理测试使用真实本地 HTTP 连接，覆盖两跳轮询、旧路径、认证与请求头替换、
OpenAI 错误格式、状态码透传、超时、重复响应头、自动解压、二进制和表单上传、
分块上传不等待完整请求体、流式数据保真、提前到达、数据库会话释放和断连清理。
第十三章独立进程检查使用模拟 Runtime，验证两个副本的就绪与轮询、OpenAI SDK 流式文本和用量、
Worker 认证，以及缩容清理后的 503 响应，不拉取模型或推理镜像。
多机测试覆盖自动与手动选卡、设备类型和状态过滤、从属资源扣减、调度结果持久化、
多 Worker 文件就绪、并发补丁与旧重启保护、主从启动协调、从属故障传播、日志轮转，
以及 Ray 和三类 MP 拓扑的主从命令、设备资源、通信环境和端口预留。
第十四章独立 Server、两台 Worker 进程检查使用 SQLite、模拟 GPU/Runtime 和本地 HTTP 引擎替身，
验证 Ray/MP 两条链路、每 Worker 的文件记录、两跳请求、整体故障恢复、从属失联重连和双方清理。
上述检查不会启动真实 Ray/vLLM 或测试 NCCL/HCCL 通信，真实多机推理需在对应设备上验收。
Kubernetes 测试覆盖多厂商清单、私有仓库 Secret、三态选项、挂载校验、旧集群表升级、
operator 订阅/取消/重试/令牌变更、TLS Unix socket、子进程异常和退出清理。
代理检查使用真实本地 HTTPS 连接，覆盖方法和状态码、重复查询和响应头、二进制、自动解压、
ServiceAccount 令牌轮换、watch 断连、异常断流和数据库会话释放。
Runtime 转 Pod 检查使用模拟节点/API，验证实际 SDK 的设备资源转换、共享内存数量和挂载继承。
第十五章独立 Server/Worker/operator 替身进程验证注册、清单、HTTPS 集群代理、订阅状态切换和 SIGTERM 清理；
未对真实 Kubernetes 集群执行 kubectl apply，也未部署真实 operator 或推理 Pod。
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
