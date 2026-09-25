"""Evidence-linked, deterministic reporting; never treats missing data as safety."""
from __future__ import annotations

import html
from typing import Any

from .models import ComponentResult

SEVERITY = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
CLOSED = {"remediated", "accepted", "resolved", "closed", "false_positive", "benign"}
LABELS = {"asset": "资产", "vulnerabilities": "漏洞", "events": "安全事件"}


def _text(value: Any, limit: int = 200) -> str:
    return " ".join(str(value).split())[:limit]


def _markdown(value: str) -> str:
    # Evidence is data, including in a Markdown viewer. Do not render its HTML,
    # links, images, headings or formatting as trusted report structure.
    value = html.escape(value)
    for character in "\\`*_{}[]()#+!|":
        value = value.replace(character, "\\" + character)
    return value


def build_report(groups: dict[str, Any]) -> dict[str, Any]:
    """Produce stable citations into the returned evidence list, not invented URLs."""
    targets = {group["ip"] for group in groups.values()}
    if len(targets) != 1:
        raise ValueError("evidence groups must refer to exactly one IP")
    ip = next(iter(targets))
    evidence = []
    findings = []
    missing = []
    for kind in LABELS:
        group = groups[kind]
        records = group["records"]
        if not records:
            missing.append(kind)
        for index, record in enumerate(records, 1):
            citation = f"E-{kind.upper()}-{index:03d}"
            severity = _text(record.get("severity", "unknown")).lower()
            if severity not in SEVERITY:
                severity = "unknown"
            state = _text(record.get("status", "unknown")).lower()
            title = _text(record.get("title") or record.get("name") or f"{LABELS[kind]}记录 {index}")
            item = {
                "citation_id": citation,
                "kind": kind,
                "source": group["source"],
                "source_record_id": _text(record.get("id") or record.get("event_id") or record.get("vulnerability_id") or record.get("asset_id") or "not_provided"),
                "ip": ip,
                "title": title,
                "severity": severity,
                "status": state,
                "verification_status": "unverified",
            }
            evidence.append(item)
            # Asset importance alone is not a vulnerability or an attack.
            if kind != "asset" and SEVERITY.get(severity, 0) > 0 and state not in CLOSED:
                findings.append({
                    "id": f"F-{len(findings) + 1:03d}",
                    "kind": kind,
                    "title": title,
                    "severity": severity,
                    "evidence_ids": [citation],
                    "verification_required": True,
                })
    findings.sort(key=lambda item: -SEVERITY[item["severity"]])
    assessment = "risk_detected" if findings else "insufficient_evidence"
    level = max((item["severity"] for item in findings), key=SEVERITY.get, default="unknown")
    limitations = [
        "输入为请求提供的未核实证据，不是实时资产库或扫描器的完整结果。",
        "风险等级仅概括输入中的有效风险记录，不代表攻击已证实或资产安全已证明。",
        "未校验来源真实性、数据新鲜度或覆盖率；本报告不是自动处置授权。",
    ]
    recommendations = (["复核引用证据和影响范围，再申请人工审批；不要直接执行封禁。"] if findings else
                       ["补充有效的资产、漏洞和事件证据；当前不能判定资产安全。"])
    if missing:
        recommendations.append("补齐数据类别：" + "、".join(LABELS[key] for key in missing) + "。")
    lines = ["# 资产安全证据报告", "", f"- 目标：{ip}",
             f"- 结论：{assessment}", f"- 已观察风险：{level}",
             f"- 证据数：{len(evidence)}", "- 分析方式：确定性规则；未调用大模型", "", "## 发现", ""]
    for finding in findings:
        lines.append(f"- {finding['id']} [{finding['severity']}] {_markdown(finding['title'])}（引用：{', '.join(finding['evidence_ids'])}；待核实）")
    if not findings:
        lines.append("没有足够证据确认风险；这不等于不存在风险。")
    lines.extend(["", "## 证据索引", ""])
    for item in evidence:
        lines.append(f"- {item['citation_id']}：{LABELS[item['kind']]} / {_markdown(item['source_record_id'])} / {_markdown(item['title'])}（未核实）")
    lines.extend(["", "## 建议", "", *[f"- {item}" for item in recommendations], "", "## 限制", "", *[f"- {item}" for item in limitations]])
    return {
        "assessment": assessment, "ip": ip, "observed_risk_level": level,
        "analysis_method": "deterministic_rules_v1", "evidence_count": len(evidence),
        "evidence": evidence, "findings": findings, "missing_evidence": missing,
        "recommendations": recommendations, "limitations": limitations,
        "report": "\n".join(lines) + "\n",
    }


def evidence_report(context):
    result = build_report(context.inputs)
    return ComponentResult(
        status="partial" if result["findings"] else "abstain",
        outputs=result, evidence=result["evidence"], warnings=result["limitations"],
    )
