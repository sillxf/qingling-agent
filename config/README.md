# 自定义配置目录

已安装包内的默认定义位于 `app/resources/config/`，包括 `agent_profile.json`、`react_profile.json`、`asset_profile.json`、`workflow_event_investigation.json`、`workflow_asset_safety.json`（1.0.0）及 `workflow_asset_safety_report.json`（1.1.0）。默认 Skill 位于 `app/resources/skills/`。不设置环境变量即可使用包内资源；根目录仅保留 `mcp_servers.example.json` 作为模板。

需要自定义时，在**项目源码目录**用 PowerShell 复制全套默认 JSON（不会改动已有数据库）：

```powershell
New-Item -ItemType Directory -Force -Path config/local | Out-Null
Copy-Item app/resources/config/*.json config/local/
$env:QINGLING_CONFIG_DIR = (Resolve-Path config/local).Path
# 如还需修改 Skill，请复制整个目录，再指定：
# Copy-Item app/resources/skills config/local/skills -Recurse
# $env:QINGLING_SKILLS_ROOT = (Resolve-Path config/local/skills).Path
& .\.venv\Scripts\python.exe -m app
```

只安装 wheel、没有源码时，可从安装位置的 `app/resources/config/` 复制 JSON；例如在所用虚拟环境中运行 `python -c "import pathlib, app; print(pathlib.Path(app.__file__).parent / 'resources' / 'config')"` 查找目录，再复制到自己可写的目录。`QINGLING_CONFIG_DIR` 和 `QINGLING_SKILLS_ROOT` 可设为绝对路径；不是 `.env` 自动加载功能。自定义配置仍需符合现有版本和契约，不能用它绕过授权与审批。

**已有数据库不会自动补入新默认项。** 启动时若存储中已有 Profile，默认导入流程不运行。演示请使用默认内存存储（不设置 `QINGLING_STORE_BACKEND`）或一个**新的**本地 SQLite 路径，例如设置 `QINGLING_STORE_BACKEND=sqlite` 和 `QINGLING_SQLITE_PATH=data/demo-new.db`；不要删除或覆盖已有库来获取新版本。若要在已有库发布新版，请走现有发布 API 并检查版本与权限。
