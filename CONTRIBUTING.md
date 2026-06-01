# 贡献指南

感谢你对 autods_agent 项目的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告问题

如果你发现了 bug 或有功能建议，请在 GitHub Issues 中创建一个新 issue，并包含以下信息：

1. **问题描述**：清晰地描述你遇到的问题
2. **复现步骤**：提供复现问题的详细步骤
3. **预期行为**：描述你期望的正确行为
4. **实际行为**：描述实际发生的情况
5. **环境信息**：Python 版本、操作系统等
6. **相关日志**：如果有的话，提供错误日志

### 提交代码

1. **Fork 项目**：在 GitHub 上 Fork 本项目
2. **克隆到本地**：
   ```bash
   git clone https://github.com/your-username/autods_agent.git
   cd autods_agent
   ```
3. **创建分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **安装依赖**：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. **进行修改**：编写代码并添加测试
6. **运行检查**：
   ```bash
   # 代码检查
   ruff check .
   
   # 格式化
   ruff format .
   
   # 类型检查
   mypy .
   
   # 运行测试
   pytest
   ```
7. **提交代码**：
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```
8. **推送到远程**：
   ```bash
   git push origin feature/your-feature-name
   ```
9. **创建 Pull Request**：在 GitHub 上创建 PR

## 代码规范

### 代码风格

- 使用 ruff 进行代码检查和格式化
- 使用 mypy 进行类型检查
- 行长度限制：88 字符
- 使用 4 空格缩进

### 类型注解

- 所有公共函数必须有完整的类型注解
- 使用 `typing` 模块中的类型
- 使用现代 Python 语法（如 `list[str]` 而不是 `List[str]`）

### 文档字符串

- 所有公共函数必须有文档字符串
- 使用 Google 风格的文档字符串
- 包含参数说明、返回值说明和异常说明

### 测试

- 所有新功能必须有对应的测试
- 测试文件命名为 `test_*.py`
- 使用 pytest 进行测试
- 测试函数命名为 `test_*`

### 提交信息

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式调整（不影响功能）
- `refactor:` 代码重构
- `test:` 添加测试
- `chore:` 其他修改（如依赖更新）

示例：
```
feat: 添加新的数据质量检查功能
fix: 修复 CSV 文件读取时的编码问题
docs: 更新 README 中的安装说明
```

## 开发环境设置

### 必需工具

- Python 3.10+
- Git
- VS Code（推荐）

### 推荐 VS Code 扩展

- Python
- Pylance
- Ruff
- GitLens

### 运行开发工具

```bash
# 安装预提交钩子
pre-commit install

# 运行所有检查
pre-commit run --all-files

# 运行特定检查
ruff check .
ruff format .
mypy .
pytest
```

## 项目结构

```
autods_agent/
├── tools/              # 工具函数模块
│   ├── data_tools.py   # 数据处理工具
│   ├── modeling_tools.py # 建模工具
│   └── utils.py        # 通用工具函数
├── tests/              # 测试目录
├── agents.py           # Agent 定义
├── run.py              # 程序入口
├── logging_config.py   # 日志配置
└── pyproject.toml      # 项目配置
```

## 行为准则

- 尊重所有参与者
- 使用包容性语言
- 接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

## 获取帮助

如果你在贡献过程中遇到问题，可以：

1. 查看项目文档
2. 在 GitHub Issues 中提问
3. 联系维护者

感谢你的贡献！