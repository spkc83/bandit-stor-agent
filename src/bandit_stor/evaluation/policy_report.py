"""Markdown policy report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_items(mapping: dict[str, Any]) -> list[str]:
    return [f"- **{k}**: {v}" for k, v in mapping.items()]


def _degeneracy_findings(ope_metrics: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if ope_metrics.get("is_near_uniform_policy"):
        findings.append(
            "- Near-uniform policy detected: entropy is near maximum and alpha divergence is near zero."
        )
    if not gates.get("checks", {}).get("avg_support_size_max", True):
        findings.append("- Sparse-support gate failed: policy support is too broad for deployment.")
    if not gates.get("checks", {}).get("alpha_divergence_min", True):
        findings.append("- Policy-movement gate failed: target policy did not move measurably from behavior.")
    if not gates.get("checks", {}).get("entropy_ratio_max", True):
        findings.append("- Entropy gate failed: target policy remains too close to maximum entropy.")
    if not gates.get("checks", {}).get("dr_lift_min", True):
        findings.append("- DR-lift gate failed: target policy did not beat logged behavior baseline.")
    return findings or ["- No explicit degeneracy finding triggered by configured gates."]


def write_policy_report(
    path: str | Path,
    *,
    dataset_summary: dict[str, Any],
    training_metrics: dict[str, Any],
    ope_metrics: dict[str, Any],
    gates: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> None:
    """Write a human-readable policy report."""
    path = Path(path)
    recommendation = "approve" if gates.get("passed") else "needs review"
    lines = [
        "# Bandit-STOR Policy Report",
        "",
        "## Dataset Summary",
        *_format_items(dataset_summary),
        "",
        "## Training Metrics",
        *_format_items(training_metrics),
        "",
        "## OPE and Diagnostics",
        *_format_items(ope_metrics),
        "",
        "## Baseline Comparison",
        *(_format_items(baseline_metrics or {})),
        "",
        "## Degeneracy Diagnostics",
        *_degeneracy_findings(ope_metrics, gates),
        "",
        "## Gate Summary",
        f"- **passed**: {gates.get('passed')}",
        *_format_items(gates.get("checks", {})),
        "",
        "## Recommendation",
        recommendation,
        "",
        "## Scope of Claims",
        "This report contains off-policy contextual-bandit estimates. DR/IPS estimates require consistency, positivity/overlap, conditional exchangeability, stable rewards, and correct or useful logged propensities. Offline OPE does not replace online A/B testing or a dedicated causal-identification analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
