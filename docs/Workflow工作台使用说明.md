# Workflow 工作台：使用与实现边界

## 启动与操作

在项目目录执行 `./scripts/start-workbench.ps1`，访问 `http://127.0.0.1:8010/workbench`。脚本默认使用 SQLite，组件清单、草稿、发布版本、启用状态、审批和运行记录保存在 `data/workbench.db`；直接启动 `python -m app` 仍沿用原有环境配置，默认内存存储。

1. 启用认证时填写 Bearer Token，点击“连接 / 刷新”。租户与角色从服务端认证主体取得。
2. 选择已有流程或新建，添加组件、配置节点、连接输出和输入端口；请求参数使用 `{"ip":{"request":"ip"}}` 绑定，常量使用 `{"ip":{"value":"10.0.0.1"}}`。配置和绑定编辑后点击“应用节点配置”。
3. 保存草稿或校验；端口冲突、缺少必填输入、Schema/语义不兼容、隐式环、不可达节点、错误配置均阻止发布。
4. 使用新版本号发布。启用版本指针用于新请求选版；“启用 / 回滚到此版”切换指针，不修改已有版本。首次没有指针时兼容旧行为，选择最高数字版本。停用后不接收新运行，已提交运行保留原版本。
5. 选择权限配置、运行租户及样例输入，执行当前发布版本。运行区展示节点状态；待审批时可批准或拒绝，也可取消任务。此处执行的是服务端已发布版本，未发布的画布修改不会生效。

`asset-safety@1.0.0` 示例并行整理三类证据。最小输入 `{"ip":"10.0.0.1"}` 会输出“证据不足”。可提供 `assets`、`vulnerabilities`、`events` 数组；每条记录须含同一租户的 `tenant_id` 和目标 `ip`。这是请求证据整理示例，不代表已经接入真实资产库或漏洞扫描器；现有 HTTP 事件源可通过 `tool.event.search` 组件使用。

`asset-safety@1.1.0` 是带引用的证据报告流程：选择已发布版本、Profile `asset-investigation` 和运行租户 `demo-tenant`；在“样例输入”粘贴 `app/resources/examples/security_cases.json` 中 `risk-with-citations.input` 的完整 JSON，再执行。合成案例会得到 `risk_detected`、3 条证据与 2 条引用发现。也可运行 `python -m app.demo --output artifacts/demo.json` 复核 9 个离线案例。输入是请求提供的未核实记录，报告并非实时扫描或真实模型研判，审批演示不执行真实封禁。

## 开发与注册

组件清单使用 `ComponentManifest`：精确 `id + version`、输入/输出/config Schema、类型、来源、维护方、依赖、超时/重试/幂等声明、权限/审批信息。Schema 属性上的 `x-semantic-type` 区分业务含义，`x-sensitivity` 阻止向低敏感级别端口传递数据。禁止同版本修改清单或发布流程。

开发者在受信任服务部署中通过 `ComponentRegistry.register(id, handler, manifest=...)` 安装函数，处理函数接收 `ComponentContext`，返回 `ComponentResult`。`context.inputs/config` 是节点专属副本，身份及取消/截止时间由 Runtime 提供；旧字典返回值仍能适配。不要让组件访问其他节点内部结构；`context.run/profile` 仅为现有内置组件保留兼容，新组件使用 `run_id/tenant_id/user_id/roles/correlation_id` 和显式输入。自定义实现需要遵守取消信号并限制外部调用时间。

工作台“注册组件”只注册引用已安装实现的清单：`implementation="组件ID@版本"`，实现契约和权限必须保持一致。上传清单不执行 Python、不自动下载插件。已安装函数、工具（含 MCP/HTTP 事件工具）、结构化模型、受控子 Agent 通过统一契约执行。禁用组件会阻止后续节点调用；重新启用不会修改清单。

JSON Schema 当前采用明确的受限方言：对象/数组/基本类型、required、enum/const、数值和长度范围、pattern、IPv4/IPv6/date-time 格式，以及上述语义标签。`$ref`、组合 Schema 等未实现关键字在注册/发布时拒绝；不声称支持完整 JSON Schema。静态校验采取保守的“源输出必须满足目标输入”规则，不能证明兼容时需要显式转换组件。

## 已接入的执行与管理能力

- 精确组件版本查找、发布清单摘要校验、Workflow 不可变发布、数字版本排序、草稿修订冲突检查、启用/停用及版本回滚。
- 有界 DAG 并发；Fork、条件 Router、Join、人工审批及有界列表 Loop。`join=any` 等待所有前驱结束后仅汇聚有效分支，不是最快分支抢占。每个输入只允许一个前驱，多路汇聚使用多个具名输入端口。
- 节点输入/输出/config 校验、默认值、显式请求绑定；节点失败终止/跳过/错误端口，partial/abstain 独立记录。失败分支和正常分支通过显式 `any` 汇聚。
- 审批绑定到运行、节点、版本、输入、配置及 Workflow 摘要，一次消费；完成节点不在审批恢复后重跑。结果不明且不幂等的中断节点拒绝自动重放。
- 节点耗时/重试次数/版本事件、运行轨迹、取消及超时、输出/状态体积上限。独立分支使用数据副本。工具及模型沿用既有运行时预算和权限边界。
- API：`GET /v1/components`，`POST /v1/components/register`，`PUT /v1/components/{id}/{version}/state`；`GET /v1/workbench/workflows`，`PUT /v1/workbench/draft`，`POST /v1/workflows/validate|publish`，`POST /v1/workbench/activate`，`POST /v1/workbench/execute/{id}/{version}`，`GET /v1/workbench/runs/{id}/trace`。

## 当前边界与后续工作

这是组件化工作台的首个可运行实现，未将原规格中的全部生产治理能力宣称为已完成：

- Loop 当前仅支持有界列表迭代，循环体限低风险、幂等的 read 函数组件；尚无任意子 Workflow 嵌套、条件退出循环、merge/reduce 或抢占式 Join。
- 组件代码仍由开发者部署；上传包隔离安装、签名验证、依赖漏洞审查、进程级沙箱和强制中止尚需专门执行环境。线程取消是协作式的，不能保证杀死忽略取消的第三方函数。
- 现有 Agent 组件调用已登记子 Agent；不等同于任意 ReAct 子流程。模型证据仅为待核验证据，不以自报置信度证明安全。
- 有实际副作用的连接器须自行实现外部幂等键、真实补偿动作及凭据服务；当前补偿声明不等于外部回滚。审批等待仍受 Run 总截止时间约束。
- 暂不支持流量百分比灰度、多人发布审批、单节点隔离调试、版本迁移向导及分布式队列租约；工作台提供样例整流程执行和节点轨迹。运行状态只适合当前单进程调度模型。
- 日志/轨迹只显示摘要；恢复所需节点数据保存在受 API 权限控制的 Run 状态中，生产环境仍需数据库加密、保留期清理及独立 artifact 存储。
