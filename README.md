# autods_agent

基于 **AutoGen AgentChat**（新版 API）的自动化数据科学分析助手课程项目：用户指定 CSV 路径与目标变量，由多个 Agent 分工完成数据概况、质量审计、基线建模、结果评估与 Markdown 报告生成。真实计算由 `tools/` 中的 Python 函数完成，LLM 只负责解读与写作。

## 功能

- 数据集概况（行/列、类型、缺失、重复、目标分布等）
- 数据质量审查（缺失、重复、常数列、类别不平衡提示）
- 自动任务类型推断（分类 / 回归）与 sklearn Pipeline 基线模型
- 5 折交叉验证指标汇总与简单最优模型规则
- 多 Agent 轮询讨论与最终结构化报告
- 完善的错误处理和输入验证
- 日志记录系统
- 单元测试和集成测试

## 项目结构

```
autods_agent/
├── .gitignore
├── .vscode/
│   └── settings.json
├── data/
│   └── titanic.csv          # 可选：默认示例路径；缺失时请自行放入数据，见「数据集准备」
├── tests/
│   ├── conftest.py
│   ├── test_agents.py
│   ├── test_data_tools.py
│   ├── test_integration.py
│   ├── test_modeling_tools.py
│   └── test_tools.py
├── tools/
│   ├── __init__.py
│   ├── data_tools.py
│   ├── modeling_tools.py
│   └── utils.py
├── agents.py
├── logging_config.py
├── pyproject.toml
├── run.py
├── requirements.txt
└── README.md
```

其中 `tools/data_tools.py` 提供 `profile_dataset` 与 `audit_dataset`；`tools/modeling_tools.py` 提供 `train_baseline_models`；`tools/utils.py` 提供通用工具函数；`agents.py` 的 `build_agents()` 创建 5 个 `AssistantAgent` 与 `RoundRobinGroupChat`；`run.py` 为异步入口并通过 `Console` 打印 `team.run_stream` 过程；`logging_config.py` 提供日志配置。

## 数据集准备

**本项目不会在运行时从网络下载数据**；请自行准备 CSV 文件。

- **默认路径**：若使用 `python run.py` 且未传 `--csv`，程序会读取项目下的 `data/titanic.csv`。若该文件不存在，`run.py` 会在启动时报错并退出。
- **放置方式**：
  1. 在项目根目录创建目录 `data/`（若尚不存在）；
  2. 将你的表格文件复制或另存为 `data/titanic.csv`（或任意文件名，配合 `--csv` 使用）；
  3. 确保文件为 UTF-8 或常见编码的可读 CSV，且包含你在 `--target` 中指定的目标列。
- **任意路径**：不依赖 `data/titanic.csv` 时，直接指定文件即可，例如  
  `python run.py --csv /你的路径/数据.csv --target 目标列名`

可从课程资料、Kaggle、或自行爬取/导出数据后按上述方式放入本地，**无需**修改代码。

## 环境安装

在项目根目录执行：

```bash
cd autods_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置 OPENAI_API_KEY

任选其一：

1. **环境变量**（推荐）

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **`.env` 文件**（与 `run.py` 同目录，已通过 `python-dotenv` 加载）

   ```
   OPENAI_API_KEY=sk-...
   ```

可选：指定模型（默认 `gpt-4o-mini`）：

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

## 运行

```bash
python run.py
```

默认使用 `./data/titanic.csv`，目标列为 `Survived`。

### 运行选项

```bash
# 指定数据集和目标列
python run.py --csv /path/to/your.csv --target 目标列名

# 设置日志级别
python run.py --log-level DEBUG

# 查看帮助
python run.py --help
```

## 更换数据集与目标变量

```bash
python run.py --csv /path/to/your.csv --target 你的目标列名
```

请确保 CSV 可读且包含所填目标列；若列不存在，工具会返回明确错误 JSON。

## 测试

### 运行所有测试

```bash
pytest
```

### 运行单元测试

```bash
pytest tests/ -v
```

### 运行集成测试

```bash
pytest tests/test_integration.py -v -m integration
```

### 生成测试覆盖率报告

```bash
pytest --cov=tools --cov=agents --cov=run --cov-report=html
```

## 代码质量检查

### 运行 ruff 检查

```bash
ruff check .
```

### 运行 ruff 格式化

```bash
ruff format .
```

### 运行 mypy 类型检查

```bash
mypy .
```

## 开发指南

### 代码风格

- 使用 ruff 进行代码检查和格式化
- 使用 mypy 进行类型检查
- 所有公共函数必须有类型注解和文档字符串
- 使用中文注释和文档

### 提交代码

1. 确保所有测试通过
2. 确保代码检查通过
3. 提交 Pull Request

## 当前版本局限性

- 无前端、无数据库、无可视化仪表盘；仅终端流式输出。
- 基线模型与特征处理较简单，未做系统调参与 AutoML。
- 多 Agent 轮询受 `MaxMessageTermination`（默认 22，兼顾工具与反思消息）与 `TERMINATE` 约束；若提前截断可在 `agents.py` 中调大。
- 依赖 OpenAI 兼容接口与网络；未内置本地大模型。
- 类别特征高基数时 OneHot 维度可能较大，未做目标编码等高级处理。

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
