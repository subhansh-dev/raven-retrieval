import numpy as np
from scipy import stats


def paired_bootstrap_test(scores_a, scores_b, n_resamples=10000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    observed_mean = np.mean(diffs)
    resampled_means = []
    for _ in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        resampled_means.append(np.mean(sample))
    resampled_means = np.array(resampled_means)
    if observed_mean > 0:
        p_value = np.mean(resampled_means <= 0)
    else:
        p_value = np.mean(resampled_means >= 0)
    ci_lower = np.percentile(resampled_means, 2.5)
    ci_upper = np.percentile(resampled_means, 97.5)
    return {
        "observed_diff": float(observed_mean),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_resamples": n_resamples,
    }


def paired_t_test(scores_a, scores_b):
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def bonferroni_correction(p_values, alpha=0.05):
    n_comparisons = len(p_values)
    adjusted_alpha = alpha / n_comparisons
    significant = [p < adjusted_alpha for p in p_values]
    return {
        "adjusted_alpha": adjusted_alpha,
        "significant": significant,
        "original_alpha": alpha,
        "n_comparisons": n_comparisons,
    }


def run_all_pairwise_tests(per_query_scores, pipeline_names, n_resamples=10000):
    n_pipelines = len(pipeline_names)
    all_p_values = []
    comparisons = []
    for i in range(n_pipelines):
        for j in range(i + 1, n_pipelines):
            name_a = pipeline_names[i]
            name_b = pipeline_names[j]
            scores_a = per_query_scores[name_a]
            scores_b = per_query_scores[name_b]
            bootstrap = paired_bootstrap_test(scores_a, scores_b, n_resamples=n_resamples)
            t_test = paired_t_test(scores_a, scores_b)
            all_p_values.append(bootstrap["p_value"])
            comparisons.append({
                "pipeline_a": name_a,
                "pipeline_b": name_b,
                "bootstrap": bootstrap,
                "t_test": t_test,
            })
    correction = bonferroni_correction(all_p_values)
    for i, comp in enumerate(comparisons):
        comp["bonferroni_significant"] = correction["significant"][i]
        comp["adjusted_alpha"] = correction["adjusted_alpha"]
    return comparisons


def format_significance_table(comparisons):
    lines = ["| Comparison | Diff | Bootstrap p | t-test p | Significant (Bonferroni) |",
             "|---|---|---|---|---|"]
    for comp in comparisons:
        sig = "Yes" if comp["bonferroni_significant"] else "No"
        line = (f"| {comp['pipeline_a']} vs {comp['pipeline_b']} | "
                f"{comp['bootstrap']['observed_diff']:.4f} | "
                f"{comp['bootstrap']['p_value']:.4f} | "
                f"{comp['t_test']['p_value']:.4f} | {sig} |")
        lines.append(line)
    return "\n".join(lines)
