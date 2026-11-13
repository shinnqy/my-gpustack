# My GPUStack：第十二章完成版

按 `guide/v1/01`～`12` 的正文及配套辅助代码整理。
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
GPUStack 入口 `/v1/chat/completions` 的代理在第十三章接入。

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
生产启动仍使用 PostgreSQL；真实 GPU/NPU 驱动采集需要在对应机器上验证。
