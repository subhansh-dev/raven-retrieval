import numpy as np
import umap
from sklearn.mixture import GaussianMixture


def reduce_dimensions(embeddings, n_neighbors=15, n_components=10):
    n_samples = embeddings.shape[0]
    if n_samples <= n_components:
        n_components = max(2, n_samples - 1)
    actual_neighbors = min(n_neighbors, n_samples - 1)
    if actual_neighbors < 2:
        actual_neighbors = 2
    reducer = umap.UMAP(
        n_neighbors=actual_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def select_cluster_count(embeddings, min_k=2, max_k=10):
    n_samples = embeddings.shape[0]
    max_k = min(max_k, n_samples - 1)
    if max_k < min_k:
        return 1
    best_k = min_k
    best_bic = float("inf")
    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k


def soft_cluster(embeddings, n_clusters=None, threshold=0.1):
    n_samples = embeddings.shape[0]
    if n_clusters is None:
        n_clusters = select_cluster_count(embeddings)
    if n_clusters <= 1 or n_samples <= 1:
        return {0: list(range(n_samples))}
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm.fit(embeddings)
    probabilities = gmm.predict_proba(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for node_idx in range(n_samples):
        for cluster_idx in range(n_clusters):
            if probabilities[node_idx, cluster_idx] > threshold:
                clusters[cluster_idx].append(node_idx)
    for k in list(clusters.keys()):
        if len(clusters[k]) == 0:
            del clusters[k]
    return clusters


def global_local_cluster(embeddings, soft_threshold=0.1):
    n_samples = embeddings.shape[0]
    if n_samples <= 2:
        return {0: list(range(n_samples))}
    global_n_neighbors = max(2, int(np.sqrt(max(n_samples - 1, 2))))
    global_reduced = reduce_dimensions(embeddings, n_neighbors=global_n_neighbors)
    global_k = select_cluster_count(global_reduced)
    global_clusters = soft_cluster(global_reduced, n_clusters=global_k, threshold=soft_threshold)
    final_clusters = {}
    final_cluster_id = 0
    for global_id, global_members in global_clusters.items():
        if len(global_members) <= 2:
            final_clusters[final_cluster_id] = global_members
            final_cluster_id += 1
            continue
        local_embeddings = embeddings[global_members]
        local_n_neighbors = min(10, len(global_members) - 1)
        if local_n_neighbors < 2:
            final_clusters[final_cluster_id] = global_members
            final_cluster_id += 1
            continue
        local_reduced = reduce_dimensions(local_embeddings, n_neighbors=max(2, local_n_neighbors))
        local_k = select_cluster_count(local_reduced)
        local_assignment = soft_cluster(local_reduced, n_clusters=local_k, threshold=soft_threshold)
        for _, local_members in local_assignment.items():
            original_indices = [global_members[i] for i in local_members]
            final_clusters[final_cluster_id] = original_indices
            final_cluster_id += 1
    return final_clusters


def compute_soft_assignment_rate(embeddings, n_clusters=None, threshold=0.1):
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        return 0.0
    if n_clusters is None:
        n_clusters = select_cluster_count(embeddings)
    if n_clusters <= 1:
        return 0.0
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm.fit(embeddings)
    probabilities = gmm.predict_proba(embeddings)
    multi_cluster_count = 0
    for node_idx in range(n_samples):
        above_threshold = sum(1 for c in range(n_clusters) if probabilities[node_idx, c] > threshold)
        if above_threshold > 1:
            multi_cluster_count += 1
    return multi_cluster_count / n_samples
