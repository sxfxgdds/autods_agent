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
from logging_config import setup_logging, get_logger
from tools.utils import validate_csv_path


def _build_task(csv_path: str, target_col: str) -> str:
    """
    构造传给团队的初始任务说明。

    Args:
        csv_path: CSV 文件路径
        target_col: 目标列名

    Returns:
        任务说明字符串
    """
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
    """
    异步主函数。

    Args:
        csv_path: CSV 文件路径
        target_col: 目标列名
    """
    logger = get_logger(__name__)
    logger.info("启动 AutoDS Agent")

    try:
        load_dotenv()
        team = build_agents()
        task = _build_task(csv_path, target_col)
        logger.info(f"开始运行分析任务: {csv_path}")
        await Console(team.run_stream(task=task))
        logger.info("分析任务完成")
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        logger.error(f"程序运行出错: {e}")
        print(f"\n错误：程序运行出错 — {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """主函数：解析命令行参数并启动程序。"""
    # 设置日志
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认 INFO）",
    )
    args = parser.parse_args()

    # 更新日志级别
    if args.log_level != "INFO":
        setup_logging(level=args.log_level)

    logger = get_logger(__name__)
    logger.info("AutoDS Agent 启动")

    # 验证 CSV 文件路径
    csv_path = os.path.abspath(args.csv)
    is_valid, error_msg = validate_csv_path(csv_path)
    if not is_valid:
        logger.error(f"CSV 文件验证失败: {error_msg}")
        print(
            f"错误：{error_msg}\n"
            "请将 CSV 放到上述路径，或使用 --csv 指定有效文件。"
            "说明见 README「数据集准备」。",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info(f"CSV 文件验证通过: {csv_path}")
    asyncio.run(_amain(csv_path, args.target))


if __name__ == "__main__":
    main()
