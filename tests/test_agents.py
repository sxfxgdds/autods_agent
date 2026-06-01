"""
Agent 构建单元测试。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agents import build_agents


class TestBuildAgents:
    """build_agents 函数测试。"""

    def test_build_agents_success(self, mock_env_vars: None) -> None:
        """测试成功构建 Agent 团队。"""
        team = build_agents()
        assert team is not None

    def test_build_agents_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试缺少 API 密钥时的错误处理。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="未设置环境变量 OPENAI_API_KEY"):
            build_agents()

    def test_build_agents_custom_model(self, mock_env_vars: None) -> None:
        """测试使用自定义模型。"""
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4"}):
            team = build_agents()
            assert team is not None
