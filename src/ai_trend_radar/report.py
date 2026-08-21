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
    language: str = "zh-CN",
) -> str:
    local_time = as_of.astimezone(ZoneInfo(timezone_name))
    english = language.casefold().startswith("en")
    copy = _REPORT_COPY["en"] if english else _REPORT_COPY["zh"]
    colon = ":" if english else "："
    lines = [
        f"# AI Trend Radar — {local_time:%Y-%m-%d}",
        "",
        f"> {copy['window']} · {copy['generated']} {local_time:%Y-%m-%d %H:%M} {timezone_name}",
        "",
    ]
    if not trends:
        lines.extend([copy["empty"], ""])
    for index, trend in enumerate(trends, 1):
        direction = _trend_direction(trend.velocity, english)
        status = copy["new"] if trend.new_count > 0 else copy["continuing"]
        confidence = (
            trend.confidence
            if english
            else {"high": "高", "medium": "中", "low": "低"}.get(
                trend.confidence.casefold(), trend.confidence
            )
        )
        lines.extend(
            [
                f"## {index}. {trend.label}",
                "",
                f"**{copy['status']}{colon} {status}** · {copy['new_count']} {trend.new_count}",
                "",
                f"**{copy['trend']}{colon} {direction}** · {copy['counts'].format(seven=trend.count_7d, thirty=trend.count_30d, sources=trend.source_count)}",
                "",
                trend.summary,
                "",
                f"**{copy['evidence_confidence']}{colon}** {trend.evidence_basis or copy['count_only']} · {confidence}",
                "",
                f"**{copy['counterevidence']}{colon}** {trend.counterevidence or copy['no_counterevidence']}",
                "",
                f"**{copy['why']}{colon}** {trend.why_it_matters}",
                "",
                f"**{copy['must_read']}{colon}**",
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
                for key, label_key in _EXPLANATION_FIELDS:
                    if explanation.get(key):
                        lines.append(f"  - **{copy[label_key]}{colon}** {explanation[key]}")
            elif explanation:
                lines.append(f"  - **{copy['method_overview']}{colon}** {explanation}")
        lines.append("")
    if warnings:
        lines.extend(["---", "", f"<details><summary>{copy['warnings']}</summary>", ""])
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
    language: str = "zh-CN",
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
        "language": language,
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
    language: str = "zh-CN",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    date = as_of.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    duration = (
        f"Target duration: about {speech.estimated_minutes} minutes"
        if language.casefold().startswith("en")
        else f"目标时长：约 {speech.estimated_minutes} 分钟"
    )
    content = (
        f"# {speech.title}\n\n"
        f"> {duration} · Language: {language} · Provider: {speech.provider}\n\n"
        f"{speech.content.strip()}\n"
    )
    path = output / f"{date}-script.md"
    path.write_text(content, encoding="utf-8")
    (output / "latest-script.md").write_text(content, encoding="utf-8")
    return path


_EXPLANATION_FIELDS = (
    ("purpose", "purpose"),
    ("approach", "approach"),
    ("difference", "difference"),
    ("evidence", "evidence"),
    ("experimental_setup", "experimental_setup"),
    ("baseline_fairness", "baseline_fairness"),
    ("ablations_and_mechanism", "ablations"),
    ("key_evidence", "key_evidence"),
    ("unproven_claims", "unproven"),
    ("limitations", "limitations"),
    ("applicability", "applicability"),
    ("adoption_prerequisites", "prerequisites"),
    ("replication_checks", "replication"),
    ("verdict", "verdict"),
    ("expert_takeaway", "expert_takeaway"),
)


_REPORT_COPY = {
    "zh": {
        "window": "过去 7/30 天信号", "generated": "生成于", "empty": "今天没有足够的相关信号形成趋势。请积累更多天的数据后重试。",
        "new": "🆕 新信号驱动", "continuing": "🔄 持续趋势", "status": "状态", "new_count": "今日首次发现",
        "trend": "趋势", "counts": "7 天 {seven} 条 / 30 天 {thirty} 条 · {sources} 类来源",
        "evidence_confidence": "证据与置信度", "count_only": "当前仅有聚类计数信号", "counterevidence": "反证 / 缺口",
        "no_counterevidence": "当前数据未提供独立反证", "why": "为什么重要", "must_read": "必读", "warnings": "采集告警",
        "purpose": "做什么", "approach": "怎么做", "difference": "有什么不同", "evidence": "实验与证据",
        "experimental_setup": "实验设置", "baseline_fairness": "基线公平性", "ablations": "消融与机制证据",
        "key_evidence": "关键证据", "unproven": "尚未证明", "limitations": "局限与证据边界", "applicability": "适用范围",
        "prerequisites": "采用前提", "replication": "复现检查", "verdict": "结论", "expert_takeaway": "技术判断", "method_overview": "方法概览",
    },
    "en": {
        "window": "7/30-day signals", "generated": "Generated at", "empty": "There are not enough related signals to form a trend today. Accumulate more history and try again.",
        "new": "🆕 New-signal driven", "continuing": "🔄 Continuing trend", "status": "Status", "new_count": "first seen today",
        "trend": "Trend", "counts": "7 days: {seven} / 30 days: {thirty} · {sources} source types",
        "evidence_confidence": "Evidence and confidence", "count_only": "Only cluster-count evidence is currently available", "counterevidence": "Counterevidence / gap",
        "no_counterevidence": "The current data provides no independent counterevidence", "why": "Why it matters", "must_read": "Must-read", "warnings": "Collection warnings",
        "purpose": "Purpose", "approach": "Approach", "difference": "What is different", "evidence": "Experiments and evidence",
        "experimental_setup": "Experimental setup", "baseline_fairness": "Baseline fairness", "ablations": "Ablations and mechanism evidence",
        "key_evidence": "Key evidence", "unproven": "Not established", "limitations": "Limitations and evidence boundary", "applicability": "Applicability",
        "prerequisites": "Adoption prerequisites", "replication": "Replication checks", "verdict": "Verdict", "expert_takeaway": "Technical judgment", "method_overview": "Method overview",
    },
}


def _trend_direction(velocity: float, english: bool) -> str:
    if english:
        return "↑↑ Rapidly rising" if velocity >= 1 else "↑ Rising" if velocity >= 0.2 else "→ Sustained activity" if velocity >= -0.2 else "↓ Cooling"
    return "↑↑ 快速升温" if velocity >= 1 else "↑ 升温" if velocity >= 0.2 else "→ 持续活跃" if velocity >= -0.2 else "↓ 回落"
