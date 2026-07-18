"""Automatic evaluation report generator.

Generates a comprehensive Markdown report from benchmark results,
including tables, significance analysis, and recommendations.

Usage:
    python -m src.eval.report --results-dir experiments/runs/enhanced_scifact_*
"""

import json
import os
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_results(results_dir):
    """Load all result files from a benchmark run directory."""
    results = {}

    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)

    sig_path = os.path.join(results_dir, "significance.json")
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            results["significance"] = json.load(f)

    timings_path = os.path.join(results_dir, "timings.json")
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            results["timings"] = json.load(f)

    meta_path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            results["metadata"] = json.load(f)

    return results


def format_ndcg_table(metrics):
    """Generate Markdown table of nDCG scores."""
    k_values = [1, 3, 5, 10, 100]
    pipelines = list(metrics.keys())

    header = "| Pipeline | " + " | ".join(f"nDCG@{k}" for k in k_values) + " |"
    sep = "|" + "|".join("---" for _ in range(len(k_values) + 1)) + "|"
    rows = [header, sep]

    # Sort by nDCG@10
    def get_ndcg10(name):
        return metrics[name].get("ndcg", {}).get("NDCG@10", 0.0)

    pipelines_sorted = sorted(pipelines, key=get_ndcg10, reverse=True)

    best_ndcg10 = get_ndcg10(pipelines_sorted[0]) if pipelines_sorted else 0

    for name in pipelines_sorted:
        cells = []
        for k in k_values:
            val = metrics[name].get("ndcg", {}).get(f"NDCG@{k}", 0.0)
            if k == 10 and val == best_ndcg10 and best_ndcg10 > 0:
                cells.append(f"**{val:.4f}**")
            else:
                cells.append(f"{val:.4f}")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def format_recall_table(metrics):
    """Generate Markdown table of Recall scores."""
    k_values = [1, 5, 10]
    pipelines = list(metrics.keys())

    header = "| Pipeline | " + " | ".join(f"Recall@{k}" for k in k_values) + " |"
    sep = "|" + "|".join("---" for _ in range(len(k_values) + 1)) + "|"
    rows = [header, sep]

    for name in pipelines:
        cells = []
        for k in k_values:
            val = metrics[name].get("recall", {}).get(f"Recall@{k}", 0.0)
            cells.append(f"{val:.4f}")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def format_significance_table(comparisons):
    """Generate Markdown significance table."""
    if not comparisons:
        return "*No significance tests run.*"

    header = "| Comparison | Δ nDCG | Bootstrap p | t-test p | Significant (Bonferroni) |"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]

    for comp in comparisons:
        sig = "✅ Yes" if comp.get("bonferroni_significant", False) else "❌ No"
        diff = comp.get("bootstrap", {}).get("observed_diff", 0)
        bp = comp.get("bootstrap", {}).get("p_value", 0)
        tp = comp.get("t_test", {}).get("p_value", 0)
        rows.append(f"| {comp['pipeline_a']} vs {comp['pipeline_b']} | "
                    f"{diff:.4f} | {bp:.4f} | {tp:.4f} | {sig} |")

    return "\n".join(rows)


def format_timings_table(timings):
    """Generate Markdown timing table."""
    if not timings:
        return "*No timing data.*"

    header = "| Pipeline | Time (s) | Relative |"
    sep = "|---|---|---|"
    rows = [header, sep]

    min_time = min(timings.values()) if timings else 1

    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        relative = t / min_time if min_time > 0 else 1
        rows.append(f"| {name} | {t:.1f}s | {relative:.1f}x |")

    return "\n".join(rows)


def generate_analysis(metrics, timings, comparisons):
    """Generate analysis text from results."""
    lines = []
    pipelines = list(metrics.keys())

    # Find best pipeline
    def get_ndcg10(name):
        return metrics[name].get("ndcg", {}).get("NDCG@10", 0.0)

    best = max(pipelines, key=get_ndcg10)
    worst = min(pipelines, key=get_ndcg10)

    lines.append(f"### Key Findings\n")
    lines.append(f"- **Best pipeline:** `{best}` with nDCG@10 = {get_ndcg10(best):.4f}")
    lines.append(f"- **Worst pipeline:** `{worst}` with nDCG@10 = {get_ndcg10(worst):.4f}")

    if get_ndcg10(best) > 0:
        improvement = ((get_ndcg10(best) - get_ndcg10(worst)) / get_ndcg10(worst)) * 100
        lines.append(f"- **Improvement:** {improvement:.1f}% over worst baseline")

    # Timing analysis
    if timings:
        fastest = min(timings, key=timings.get)
        slowest = max(timings, key=timings.get)
        lines.append(f"\n### Timing\n")
        lines.append(f"- **Fastest:** `{fastest}` ({timings[fastest]:.1f}s)")
        lines.append(f"- **Slowest:** `{slowest}` ({timings[slowest]:.1f}s)")
        if timings[fastest] > 0:
            speedup = timings[slowest] / timings[fastest]
            lines.append(f"- **Speedup range:** {speedup:.1f}x between fastest and slowest")

    # Significance analysis
    if comparisons:
        sig_count = sum(1 for c in comparisons if c.get("bonferroni_significant", False))
        lines.append(f"\n### Statistical Significance\n")
        lines.append(f"- {sig_count}/{len(comparisons)} pairwise comparisons are significant (Bonferroni corrected)")
        if sig_count == 0:
            lines.append(f"- No pairs reach significance — likely due to small sample size or similar performance")

    return "\n".join(lines)


def generate_report(results_dir, output_path=None):
    """Generate a comprehensive Markdown evaluation report.

    Args:
        results_dir: path to benchmark results directory
        output_path: where to write the report (default: results_dir/REPORT.md)
    """
    results = load_results(results_dir)

    if not results.get("metrics"):
        logger.error(f"No metrics found in {results_dir}")
        return None

    if output_path is None:
        output_path = os.path.join(results_dir, "REPORT.md")

    metrics = results["metrics"]
    timings = results.get("timings", {})
    comparisons = results.get("significance", [])
    metadata = results.get("metadata", {})

    # Build report
    sections = []

    sections.append("# Benchmark Report\n")
    sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"**Results directory:** `{results_dir}`")
    if metadata.get("run_id"):
        sections.append(f"**Run ID:** `{metadata['run_id']}`")
    sections.append("")

    sections.append("## Summary\n")
    sections.append(format_ndcg_table(metrics))
    sections.append("")

    sections.append("## Recall\n")
    sections.append(format_recall_table(metrics))
    sections.append("")

    if timings:
        sections.append("## Latency\n")
        sections.append(format_timings_table(timings))
        sections.append("")

    if comparisons:
        sections.append("## Statistical Significance\n")
        sections.append(format_significance_table(comparisons))
        sections.append("")

    sections.append("## Analysis\n")
    sections.append(generate_analysis(metrics, timings, comparisons))
    sections.append("")

    sections.append("## Recommendations\n")

    # Generate recommendations based on results
    pipelines = list(metrics.keys())
    def get_ndcg10(name):
        return metrics[name].get("ndcg", {}).get("NDCG@10", 0.0)

    best = max(pipelines, key=get_ndcg10)
    fastest = min(timings, key=timings.get) if timings else None

    sections.append("### For Production\n")
    if fastest and best:
        if fastest == best:
            sections.append(f"- `{best}` is both the best and fastest — use it directly")
        else:
            sections.append(f"- **Best quality:** `{best}` (nDCG@10={get_ndcg10(best):.4f})")
            sections.append(f"- **Best speed:** `{fastest}` ({timings[fastest]:.1f}s)")
            sections.append(f"- Consider `{fastest}` for latency-sensitive applications")

    sections.append("\n### For Research\n")
    sections.append("- Fine-tune ColBERT projection layer on retrieval triples")
    sections.append("- Test RAPTOR + Late Interaction on multi-hop datasets (HotpotQA)")
    sections.append("- Compare SPLADE vs BM25 for term expansion effectiveness")
    sections.append("- Evaluate HyDE with larger generator models")

    report = "\n".join(sections)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("results_dir", help="Path to benchmark results directory")
    parser.add_argument("--output", default=None, help="Output path (default: results_dir/REPORT.md)")
    args = parser.parse_args()

    report_path = generate_report(args.results_dir, args.output)
    if report_path:
        print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
