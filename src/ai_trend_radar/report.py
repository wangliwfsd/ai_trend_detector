from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import SpeechScript, Trend


def render_markdown(
    trends: list[Trend],
    as_of: datetime,
    timezone_name: str,
    must_reads: int = 2,
    warnings: list[str] | None = None,
) -> str:
    local_time = as_of.astimezone(ZoneInfo(timezone_name))
    lines = [
        f"# AI Trend Radar — {local_time:%Y-%m-%d}",
        "",
        f"> 过去 7/30 天信号 · 生成于 {local_time:%Y-%m-%d %H:%M} {timezone_name}",
        "",
    ]
    if not trends:
        lines.extend(["今天没有足够的相关信号形成趋势。请积累更多天的数据后重试。", ""])
    for index, trend in enumerate(trends, 1):
        direction = "↑↑ 快速升温" if trend.velocity >= 1 else "↑ 升温" if trend.velocity >= 0.2 else "→ 持续活跃" if trend.velocity >= -0.2 else "↓ 回落"
        status = "🆕 新信号驱动" if trend.new_count > 0 else "🔄 持续趋势"
        lines.extend(
            [
                f"## {index}. {trend.label}",
                "",
                f"**状态：{status}** · 今日首次发现 {trend.new_count} 条",
                "",
                f"**趋势：{direction}** · 7 天 {trend.count_7d} 条 / 30 天 {trend.count_30d} 条 · {trend.source_count} 类来源",
                "",
                trend.summary,
                "",
                f"**证据与置信度：** {trend.evidence_basis or '当前仅有聚类计数信号'} · {trend.confidence}",
                "",
                f"**反证 / 缺口：** {trend.counterevidence or '当前数据未提供独立反证'}",
                "",
                f"**为什么重要：** {trend.why_it_matters}",
                "",
                "**必读：**",
                "",
            ]
        )
        seen: set[str] = set()
        selected = []
        for item in trend.items:
            if item.url in seen:
                continue
            seen.add(item.url)
            selected.append(item)
            if len(selected) >= must_reads:
                break
        for item in selected:
            lines.append(f"- [{item.title}]({item.url}) — {item.source}")
            explanation = item.metadata.get("method_explanation")
            if isinstance(explanation, dict):
                lines.append(f"  - **做什么：** {explanation.get('purpose', '')}")
                lines.append(f"  - **怎么做：** {explanation.get('approach', '')}")
                lines.append(f"  - **有什么不同：** {explanation.get('difference', '')}")
                if explanation.get("evidence"):
                    lines.append(f"  - **实验与证据：** {explanation['evidence']}")
                if explanation.get("experimental_setup"):
                    lines.append(f"  - **实验设置：** {explanation['experimental_setup']}")
                if explanation.get("baseline_fairness"):
                    lines.append(f"  - **基线公平性：** {explanation['baseline_fairness']}")
                if explanation.get("ablations_and_mechanism"):
                    lines.append(f"  - **消融与机制证据：** {explanation['ablations_and_mechanism']}")
                if explanation.get("key_evidence"):
                    lines.append(f"  - **关键证据：** {explanation['key_evidence']}")
                if explanation.get("unproven_claims"):
                    lines.append(f"  - **尚未证明：** {explanation['unproven_claims']}")
                if explanation.get("limitations"):
                    lines.append(f"  - **局限与证据边界：** {explanation['limitations']}")
                if explanation.get("applicability"):
                    lines.append(f"  - **适用范围：** {explanation['applicability']}")
                if explanation.get("adoption_prerequisites"):
                    lines.append(f"  - **采用前提：** {explanation['adoption_prerequisites']}")
                if explanation.get("replication_checks"):
                    lines.append(f"  - **复现检查：** {explanation['replication_checks']}")
                if explanation.get("verdict"):
                    lines.append(f"  - **结论：** {explanation['verdict']}")
                if explanation.get("expert_takeaway"):
                    lines.append(f"  - **技术判断：** {explanation['expert_takeaway']}")
            elif explanation:
                lines.append(f"  - **方法概览：** {explanation}")
        lines.append("")
    if warnings:
        lines.extend(["---", "", "<details><summary>采集告警</summary>", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.extend(["", "</details>", ""])
    return "\n".join(lines).strip() + "\n"


def write_reports(
    output_dir: str | Path,
    markdown: str,
    trends: list[Trend],
    as_of: datetime,
    timezone_name: str,
    must_reads: int = 2,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    date = as_of.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    markdown_path = output / f"{date}.md"
    json_path = output / f"{date}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    payload = {
        "date": date,
        "generated_at": as_of.isoformat(),
        "trends": [
            {
                "label": trend.label,
                "score": round(trend.score, 4),
                "velocity": round(trend.velocity, 4),
                "count_7d": trend.count_7d,
                "count_30d": trend.count_30d,
                "source_count": trend.source_count,
                "status": "new_signals" if trend.new_count > 0 else "continuing",
                "new_count": trend.new_count,
                "summary": trend.summary,
                "why_it_matters": trend.why_it_matters,
                "evidence_basis": trend.evidence_basis,
                "confidence": trend.confidence,
                "counterevidence": trend.counterevidence,
                "must_reads": [
                    {
                        "title": item.title,
                        "url": item.url,
                        "source": item.source,
                        "method_explanation": item.metadata.get("method_explanation", ""),
                    }
                    for item in trend.items[:must_reads]
                ],
            }
            for trend in trends
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "latest.md").write_text(markdown, encoding="utf-8")
    return markdown_path, json_path


def write_speech_script(
    output_dir: str | Path,
    speech: SpeechScript,
    as_of: datetime,
    timezone_name: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    date = as_of.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    content = (
        f"# {speech.title}\n\n"
        f"> 目标时长：约 {speech.estimated_minutes} 分钟 · Provider: {speech.provider}\n\n"
        f"{speech.content.strip()}\n"
    )
    path = output / f"{date}-script.md"
    path.write_text(content, encoding="utf-8")
    (output / "latest-script.md").write_text(content, encoding="utf-8")
    return path
