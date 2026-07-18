"""Benchmark visualization and dashboard generation.

Generates:
- Bar charts comparing nDCG across pipelines
- Radar plots for multi-metric comparison
- Latency vs. quality tradeoff plots
- Statistical significance heatmaps
- Summary HTML dashboard

Output: self-contained HTML file with embedded SVG charts.
"""

import json
import os
import math
import logging

logger = logging.getLogger(__name__)


def _svg_bar_chart(labels, values, title="", width=600, height=300, color="#4A90D9"):
    """Generate SVG bar chart."""
    if not labels:
        return ""

    max_val = max(values) if values else 1
    bar_width = max(20, min(80, (width - 100) // len(labels)))
    gap = max(5, bar_width // 4)
    chart_height = height - 80
    chart_width = len(labels) * (bar_width + gap)

    svg_width = max(width, chart_width + 100)

    bars = []
    texts = []
    for i, (label, val) in enumerate(zip(labels, values)):
        x = 50 + i * (bar_width + gap)
        bar_h = (val / max_val) * chart_height if max_val > 0 else 0
        y = chart_height - bar_h + 20

        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" opacity="0.85" rx="3"/>')
        bars.append(f'<text x="{x + bar_width/2}" y="{chart_height + 40}" text-anchor="middle" font-size="11" fill="#333">{label[:15]}</text>')
        bars.append(f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="10" fill="#555">{val:.3f}</text>')

    svg = f'''<svg width="{svg_width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <text x="{svg_width/2}" y="15" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    {"".join(bars)}
</svg>'''
    return svg


def _svg_grouped_bar_chart(labels, groups, group_labels, title="", width=700, height=350):
    """Generate SVG grouped bar chart for multi-pipeline comparison."""
    if not labels or not groups:
        return ""

    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
              "#E67E22", "#3498DB", "#E91E63", "#00BCD4"]

    n_groups = len(labels)
    n_bars = len(groups)
    bar_width = max(15, min(40, (width - 150) // (n_groups * n_bars)))
    gap = max(3, bar_width // 3)
    group_gap = max(10, bar_width)
    chart_height = height - 100

    max_val = max(max(g) for g in groups) if groups else 1
    total_width = n_groups * (n_bars * (bar_width + gap) + group_gap)

    svg_width = max(width, total_width + 150)

    elements = []
    for gi, (group_vals, gl) in enumerate(zip(groups, group_labels)):
        color = colors[gi % len(colors)]
        for li, val in enumerate(group_vals):
            x = 70 + li * (n_bars * (bar_width + gap) + group_gap) + gi * (bar_width + gap)
            bar_h = (val / max_val) * chart_height if max_val > 0 else 0
            y = chart_height - bar_h + 30

            elements.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" opacity="0.85" rx="2"/>')

    # X-axis labels
    for li, label in enumerate(labels):
        x = 70 + li * (n_bars * (bar_width + gap) + group_gap) + (n_bars * (bar_width + gap)) / 2 - gap
        elements.append(f'<text x="{x}" y="{chart_height + 55}" text-anchor="middle" font-size="10" fill="#333" transform="rotate(-30,{x},{chart_height + 55})">{label[:20]}</text>')

    # Legend
    for gi, gl in enumerate(group_labels):
        color = colors[gi % len(colors)]
        y_pos = height - 20
        x_pos = 70 + gi * 120
        elements.append(f'<rect x="{x_pos}" y="{y_pos - 8}" width="10" height="10" fill="{color}" opacity="0.85"/>')
        elements.append(f'<text x="{x_pos + 14}" y="{y_pos}" font-size="10" fill="#333">{gl[:15]}</text>')

    svg = f'''<svg width="{svg_width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <text x="{svg_width/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    {"".join(elements)}
</svg>'''
    return svg


def _svg_radar_chart(metrics_dict, title="", width=400, height=400):
    """Generate SVG radar/spider chart for multi-metric comparison.

    metrics_dict: {pipeline_name: {metric_name: value}}
    """
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    cx, cy = width // 2, height // 2 + 10
    radius = min(cx, cy) - 60

    # Get all metric names
    all_metrics = set()
    for v in metrics_dict.values():
        all_metrics.update(v.keys())
    metrics = sorted(all_metrics)
    n = len(metrics)

    if n < 3:
        return ""

    angle_step = 2 * math.pi / n

    elements = []

    # Draw grid circles
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = radius * r_frac
        points = []
        for i in range(n):
            angle = -math.pi / 2 + i * angle_step
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append(f"{x},{y}")
        elements.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="#ddd" stroke-width="0.5"/>')

    # Draw axes
    for i in range(n):
        angle = -math.pi / 2 + i * angle_step
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        elements.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#ccc" stroke-width="0.5"/>')

        # Labels
        lx = cx + (radius + 20) * math.cos(angle)
        ly = cy + (radius + 20) * math.sin(angle)
        anchor = "middle"
        if lx < cx - 10:
            anchor = "end"
        elif lx > cx + 10:
            anchor = "start"
        elements.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="10" fill="#333" dominant-baseline="middle">{metrics[i]}</text>')

    # Draw data polygons
    for pi, (name, vals) in enumerate(metrics_dict.items()):
        color = colors[pi % len(colors)]
        max_val = max(vals.values()) if vals else 1
        points = []
        for i in range(n):
            val = vals.get(metrics[i], 0)
            r = (val / max_val) * radius if max_val > 0 else 0
            angle = -math.pi / 2 + i * angle_step
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append(f"{x},{y}")

        elements.append(f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>')

    # Legend
    for pi, name in enumerate(metrics_dict.keys()):
        color = colors[pi % len(colors)]
        y_pos = height - 15 + (pi // 3) * 15
        x_pos = 20 + (pi % 3) * 140
        elements.append(f'<rect x="{x_pos}" y="{y_pos - 8}" width="10" height="10" fill="{color}" opacity="0.85"/>')
        elements.append(f'<text x="{x_pos + 14}" y="{y_pos}" font-size="10" fill="#333">{name[:18]}</text>')

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <text x="{width/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    {"".join(elements)}
</svg>'''
    return svg


def _svg_scatter_plot(points, title="", xlabel="", ylabel="", width=500, height=350):
    """SVG scatter plot for latency vs quality tradeoff.

    points: [(x, y, label, color), ...]
    """
    if not points:
        return ""

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1

    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    elements = []

    # Axes
    elements.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>')
    elements.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="1"/>')

    # Points
    for x, y, label, color in points:
        px = margin + ((x - min_x) / range_x) * plot_w
        py = height - margin - ((y - min_y) / range_y) * plot_h
        elements.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{color}" opacity="0.7"/>')
        elements.append(f'<text x="{px + 10}" y="{py - 5}" font-size="9" fill="#333">{label[:15]}</text>')

    # Labels
    elements.append(f'<text x="{width/2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#555">{xlabel}</text>')
    elements.append(f'<text x="15" y="{height/2}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90,15,{height/2})">{ylabel}</text>')

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <text x="{width/2}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    {"".join(elements)}
</svg>'''
    return svg


def generate_dashboard(all_metrics, pipeline_timings=None, significance_results=None,
                       output_path="benchmark_dashboard.html"):
    """Generate a comprehensive HTML dashboard with SVG charts.

    Args:
        all_metrics: {pipeline_name: {"ndcg": {...}, "map": {...}, ...}}
        pipeline_timings: {pipeline_name: seconds}
        significance_results: list of pairwise comparison dicts
        output_path: where to write the HTML
    """
    pipeline_names = list(all_metrics.keys())
    k_values = [1, 3, 5, 10]

    # Extract nDCG values
    ndcg_data = {}
    for name in pipeline_names:
        ndcg_data[name] = {}
        for k in k_values:
            ndcg_data[name][k] = all_metrics[name]["ndcg"].get(f"NDCG@{k}", 0.0)

    # --- nDCG@10 bar chart ---
    ndcg10_labels = [p[:15] for p in pipeline_names]
    ndcg10_values = [ndcg_data[p][10] for p in pipeline_names]
    ndcg10_chart = _svg_bar_chart(ndcg10_labels, ndcg10_values,
                                  title="nDCG@10 by Pipeline", width=700, height=300)

    # --- Grouped bar chart: nDCG@k ---
    groups = []
    for k in k_values:
        groups.append([ndcg_data[p][k] for p in pipeline_names])
    grouped_chart = _svg_grouped_bar_chart(
        [p[:15] for p in pipeline_names], groups,
        [f"nDCG@{k}" for k in k_values],
        title="nDCG@k Comparison", width=800, height=350
    )

    # --- Radar chart ---
    radar_data = {}
    for name in pipeline_names:
        radar_data[name[:15]] = {
            f"nDCG@{k}": ndcg_data[name][k] for k in k_values
        }
        # Add recall if available
        for rk in [1, 5, 10]:
            key = f"Recall@{rk}"
            val = all_metrics[name].get("recall", {}).get(f"Recall@{rk}", None)
            if val is not None:
                radar_data[name[:15]][key] = val

    radar_chart = _svg_radar_chart(radar_data, title="Multi-Metric Radar", width=450, height=420)

    # --- Latency vs Quality scatter ---
    scatter_chart = ""
    if pipeline_timings:
        colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
        scatter_points = []
        for i, name in enumerate(pipeline_names):
            timing = pipeline_timings.get(name, 0)
            quality = ndcg_data[name].get(10, 0)
            scatter_points.append((timing, quality, name[:15], colors[i % len(colors)]))
        scatter_chart = _svg_scatter_plot(
            scatter_points,
            title="Latency vs. Quality Tradeoff",
            xlabel="Total Time (s)", ylabel="nDCG@10",
            width=550, height=350
        )

    # --- Significance table ---
    sig_html = ""
    if significance_results:
        rows = []
        for comp in significance_results:
            sig_class = "sig-yes" if comp.get("bonferroni_significant", False) else "sig-no"
            rows.append(f'''<tr>
                <td>{comp["pipeline_a"][:20]} vs {comp["pipeline_b"][:20]}</td>
                <td>{comp.get("bootstrap", {}).get("observed_diff", 0):.4f}</td>
                <td>{comp.get("bootstrap", {}).get("p_value", 0):.4f}</td>
                <td class="{sig_class}">{"Yes" if comp.get("bonferroni_significant") else "No"}</td>
            </tr>''')

        sig_html = f'''
        <h2>Statistical Significance</h2>
        <table>
            <tr><th>Comparison</th><th>Δ nDCG</th><th>Bootstrap p</th><th>Significant</th></tr>
            {"".join(rows)}
        </table>'''

    # --- Summary table ---
    summary_rows = []
    for name in pipeline_names:
        cells = "".join(
            f'<td>{ndcg_data[name].get(k, 0):.4f}</td>'
            for k in k_values
        )
        timing = f'{pipeline_timings.get(name, 0):.1f}s' if pipeline_timings else "—"
        summary_rows.append(f'<tr><td><strong>{name}</strong></td>{cells}<td>{timing}</td></tr>')

    summary_table = f'''
    <table>
        <tr><th>Pipeline</th>{"".join(f"<th>nDCG@{k}</th>" for k in k_values)}<th>Time</th></tr>
        {"".join(summary_rows)}
    </table>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raven-Retrieval Benchmark Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f7fa; color: #333; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; margin: 20px 0; color: #2c3e50; }}
        h2 {{ color: #34495e; margin: 30px 0 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .chart-row {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin: 20px 0; }}
        .chart-card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 15px 0; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .sig-yes {{ color: #27ae60; font-weight: bold; }}
        .sig-no {{ color: #95a5a6; }}
        .meta {{ text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐦‍⬛ Raven-Retrieval Benchmark Dashboard</h1>

        <h2>Summary</h2>
        {summary_table}

        <h2>nDCG@10 Comparison</h2>
        <div class="chart-card">{ndcg10_chart}</div>

        <h2>nDCG@k Breakdown</h2>
        <div class="chart-card">{grouped_chart}</div>

        <div class="chart-row">
            <div class="chart-card">{radar_chart}</div>
            <div class="chart-card">{scatter_chart}</div>
        </div>

        {sig_html}

        <p class="meta">Generated by Raven-Retrieval Benchmark Suite</p>
    </div>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")
    return output_path
