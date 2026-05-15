"""
入口：构造任务并运行多 Agent 团队（AgentChat + RoundRobin）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autogen_agentchat.ui import Console
from dotenv import load_dotenv

from agents import build_agents


def _build_task(csv_path: str, target_col: str) -> str:
    """构造传给团队的初始任务说明。"""
    return (
        "请协作完成自动化数据科学分析任务。\n"
        f"- 数据集 CSV 路径：{csv_path}\n"
        f"- 目标变量列名：{target_col}\n\n"
        "要求：\n"
        "1) DataAnalystAgent 先使用 profile_dataset 工具；\n"
        "2) DataAuditorAgent 使用 audit_dataset 工具；\n"
        "3) ModelingAgent 使用 train_baseline_models 工具；\n"
        "4) EvaluationAgent 基于前述工具输出做风险与改进评估；\n"
        "5) ReportAgent 生成规定结构的 Markdown 报告，并在最后一行输出 TERMINATE。\n"
        "所有统计与模型指标必须以工具返回的 JSON 为准，不得编造。"
    )


async def _amain(csv_path: str, target_col: str) -> None:
    load_dotenv()
    team = build_agents()
    task = _build_task(csv_path, target_col)
    await Console(team.run_stream(task=task))


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDS Agent — 自动化数据科学分析助手")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "data", "titanic.csv"),
        help="CSV 数据集路径（默认 ./data/titanic.csv）",
    )
    parser.add_argument(
        "--target",
        default="Survived",
        help="目标变量列名（默认 Survived）",
    )
    args = parser.parse_args()
    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(
            f"错误：找不到数据文件：{csv_path}\n"
            "请将 CSV 放到上述路径，或使用 --csv 指定有效文件。"
            "说明见 README「数据集准备」。",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(_amain(csv_path, args.target))


if __name__ == "__main__":
    main()
