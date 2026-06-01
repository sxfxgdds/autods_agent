"""
多 Agent 协作：基于 AutoGen AgentChat（AssistantAgent + RoundRobinGroupChat）。
"""

from __future__ import annotations

import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from logging_config import get_logger
from tools.data_tools import audit_dataset, profile_dataset
from tools.modeling_tools import train_baseline_models

logger = get_logger(__name__)


def _create_model_client() -> OpenAIChatCompletionClient:
    """
    创建 OpenAI 模型客户端。

    Returns:
        OpenAIChatCompletionClient 实例

    Raises:
        RuntimeError: 如果未设置 OPENAI_API_KEY 环境变量
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置环境变量 OPENAI_API_KEY。请在 .env 或 shell 中配置后再运行。"
        )

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    logger.info(f"创建模型客户端: {model_name}")

    return OpenAIChatCompletionClient(model=model_name, api_key=api_key)


def _create_data_analyst_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """
    创建数据分析 Agent。

    Args:
        model_client: 模型客户端

    Returns:
        AssistantAgent 实例
    """
    return AssistantAgent(
        "DataAnalystAgent",
        model_client=model_client,
        tools=[profile_dataset],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
        description="负责调用 profile_dataset 并解读数据集概况。",
        system_message=(
            "你是 DataAnalystAgent。你的职责：\n"
            "1) 必须调用工具 profile_dataset(csv_path, target_col) 获取数据概况；\n"
            "2) 仅根据工具返回的 JSON 总结行数、列数、字段类型、数值/类别列、缺失、重复行、目标变量分布；\n"
            "3) 根据工具结果判断更适合分类还是回归（与后续建模规则一致："
            "非数值目标或数值目标唯一值<=20 视为分类）；\n"
            "4) 严禁编造工具未给出的统计量或列名。\n"
            "用中文简洁回复，并在结尾提示团队可继续由数据审计员分析。"
        ),
    )


def _create_data_auditor_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """
    创建数据审计 Agent。

    Args:
        model_client: 模型客户端

    Returns:
        AssistantAgent 实例
    """
    return AssistantAgent(
        "DataAuditorAgent",
        model_client=model_client,
        tools=[audit_dataset],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
        description="负责调用 audit_dataset 并给出数据质量与清洗建议。",
        system_message=(
            "你是 DataAuditorAgent。你的职责：\n"
            "1) 必须调用工具 audit_dataset(csv_path, target_col)；\n"
            "2) 根据返回的 issues、suggestions 说明缺失值、重复样本、常数列与目标类别不平衡情况；\n"
            "3) 给出可执行的数据清洗与特征工程建议；\n"
            "4) 不得臆造工具未列出的问题。\n"
            "用中文简洁回复。"
        ),
    )


def _create_modeling_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """
    创建建模 Agent。

    Args:
        model_client: 模型客户端

    Returns:
        AssistantAgent 实例
    """
    return AssistantAgent(
        "ModelingAgent",
        model_client=model_client,
        tools=[train_baseline_models],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
        description="负责调用 train_baseline_models 并解释交叉验证结果。",
        system_message=(
            "你是 ModelingAgent。你的职责：\n"
            "1) 必须调用工具 train_baseline_models(csv_path, target_col)；\n"
            "2) 说明 task_type、样本数、删除的 ID 类字段、训练了哪些模型；\n"
            "3) 解释 5 折交叉验证指标（分类：accuracy、f1_weighted，二分类还有 roc_auc；"
            "回归：MAE、RMSE、R2），并指出 best_model_by_rule 对应的最优模型；\n"
            "4) 不得编造指标，所有数字须来自工具 JSON。\n"
            "用中文简洁回复。"
        ),
    )


def _create_evaluation_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """
    创建评估 Agent。

    Args:
        model_client: 模型客户端

    Returns:
        AssistantAgent 实例
    """
    return AssistantAgent(
        "EvaluationAgent",
        model_client=model_client,
        tools=[],
        description="审查前面分析，提示风险与改进方向。",
        system_message=(
            "你是 EvaluationAgent。你没有工具调用权限。\n"
            "基于对话中前面 Agent 已给出的工具 JSON 结果（概况、审计、建模），审查：\n"
            "- 是否存在类别不平衡及其对指标的影响；\n"
            "- 指标是否可能虚高/过拟合（样本量、特征数、交叉验证方差等）；\n"
            "- 基线模型局限性；\n"
            "给出改进建议。不得捏造未在对话中出现的数据。\n"
            "用中文简洁回复。"
        ),
    )


def _create_report_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """
    创建报告 Agent。

    Args:
        model_client: 模型客户端

    Returns:
        AssistantAgent 实例
    """
    return AssistantAgent(
        "ReportAgent",
        model_client=model_client,
        tools=[],
        description="汇总生成 Markdown 报告并以 TERMINATE 结束。",
        system_message=(
            "你是 ReportAgent。你没有工具调用权限。\n"
            "根据整轮对话中各 Agent 提供的、来自工具的真实结果，生成一份 **Markdown** 报告。\n"
            "报告结构必须严格包含以下标题（使用 Markdown 一级/二级标题）：\n"
            "# 基于 AutoGen 的自动化数据科学分析报告\n"
            "## 1. 数据集概况\n"
            "## 2. 数据质量分析\n"
            "## 3. 建模方法\n"
            "## 4. 实验结果\n"
            "## 5. 结果讨论\n"
            "## 6. 系统优势与不足\n"
            "## 7. 后续改进方向\n"
            "内容须与前序分析一致，不得虚构指标。\n"
            "在报告正文之后，**单独一行**输出单词：TERMINATE\n"
            "（该行仅包含 TERMINATE，表示整个团队任务结束。）"
        ),
    )


def build_agents() -> RoundRobinGroupChat:
    """
    构建模型客户端与 5 个 Agent，并返回配置好终止条件的 RoundRobinGroupChat。

    Returns:
        配置好的 RoundRobinGroupChat 实例
    """
    logger.info("开始构建 Agent 团队")

    try:
        model_client = _create_model_client()

        data_analyst = _create_data_analyst_agent(model_client)
        data_auditor = _create_data_auditor_agent(model_client)
        modeling = _create_modeling_agent(model_client)
        evaluation = _create_evaluation_agent(model_client)
        report = _create_report_agent(model_client)

        # 课程建议 12–15 条消息；含工具调用时略放宽，避免在 Report 前被截断
        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(22)

        team = RoundRobinGroupChat(
            [
                data_analyst,
                data_auditor,
                modeling,
                evaluation,
                report,
            ],
            termination_condition=termination,
        )

        logger.info("Agent 团队构建完成")
        return team

    except Exception as e:  # noqa: BLE001
        logger.error(f"构建 Agent 团队时出错: {e}")
        raise
