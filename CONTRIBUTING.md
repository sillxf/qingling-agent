# 参与开发

1. 先阅读 [README](README.md)、[文档索引](docs/README.md) 与 [安全说明](SECURITY.md)。不要在 issue、截图、测试夹具或 PR 中提交真实告警、token、数据库或个人信息。
2. 使用 Python 3.10+ 创建虚拟环境并安装 `pip install -c requirements/constraints.txt -e '.[dev]'`。修改前确认现有测试与演示；尽量提交小而可验证的变更，不修改已发布 Workflow / 组件的同版本内容。
3. 提交前运行 `python -m ruff check app tests scripts`、`python -m pytest -q`、`python -m app.demo --output artifacts/demo.json`，并说明环境与结果。演示输出是合成数据，不是模型准确率证明。
4. 涉及租户、认证、授权、审批、证据结论语义、持久化或 API 契约时，先说明风险和回归计划；不能为使测试通过而关闭安全检查。报告漏洞请按 [SECURITY.md](SECURITY.md) 私下联系维护者。

本项目以 [MIT License](LICENSE) 授权，现有项目代码版权归 sillxf。提交贡献时请确认拥有提交内容的授权，并保留适用的版权及许可声明。
