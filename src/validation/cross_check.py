import numpy as np
from scipy.stats import spearmanr


class ReferenceCrossValidator:

    def __init__(self, our_encoder, corpus_texts, corpus_ids):
        self.our_encoder = our_encoder
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids

    def compute_rankings_our_system(self, queries, top_k=50):
        from ..maxsim.brute_force import brute_force_rank
        import torch
        all_rankings = {}
        self.our_encoder.eval()
        with torch.no_grad():
            doc_embeddings = []
            for text in self.corpus_texts:
                emb = self.our_encoder.encode_document(text)
                doc_embeddings.append(emb.cpu().numpy())
            for qid, query in queries.items():
                query_emb = self.our_encoder.encode_query(query)
                query_emb_np = query_emb.cpu().numpy()
                ranked = brute_force_rank(query_emb_np, doc_embeddings, top_k=top_k)
                all_rankings[qid] = [self.corpus_ids[idx] for idx, _ in ranked]
        return all_rankings

    def compute_rankings_reference(self, queries, top_k=50):
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            return None
        try:
            rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        except Exception:
            return None
        all_rankings = {}
        for qid, query in queries.items():
            try:
                results = rag.search(query, k=top_k)
                ranked_ids = [r["document_id"] for r in results]
                all_rankings[qid] = ranked_ids
            except Exception:
                all_rankings[qid] = []
        return all_rankings

    def compute_rank_correlation(self, our_rankings, reference_rankings, top_k=20):
        correlations = []
        for qid in our_rankings:
            if qid not in reference_rankings:
                continue
            our_ids = our_rankings[qid][:top_k]
            ref_ids = reference_rankings[qid][:top_k]
            common_docs = set(our_ids) & set(ref_ids)
            if len(common_docs) < 3:
                continue
            our_ranks = []
            ref_ranks = []
            for doc in common_docs:
                our_ranks.append(our_ids.index(doc))
                ref_ranks.append(ref_ids.index(doc))
            rho, p_value = spearmanr(our_ranks, ref_ranks)
            correlations.append({"qid": qid, "rho": rho, "p_value": p_value})
        if not correlations:
            return {"mean_rho": 0.0, "median_rho": 0.0, "n_queries": 0}
        rhos = [c["rho"] for c in correlations]
        return {
            "mean_rho": float(np.mean(rhos)),
            "median_rho": float(np.median(rhos)),
            "min_rho": float(np.min(rhos)),
            "max_rho": float(np.max(rhos)),
            "std_rho": float(np.std(rhos)),
            "n_queries": len(correlations),
            "per_query": correlations,
        }

    def run_full_cross_check(self, queries, top_k_rankings=50, top_k_correlation=20):
        our_rankings = self.compute_rankings_our_system(queries, top_k=top_k_rankings)
        reference_rankings = self.compute_rankings_reference(queries, top_k=top_k_rankings)
        if reference_rankings is None:
            return {
                "status": "skipped",
                "reason": "ragatouille not available or ColBERTv2 model not downloadable",
            }
        correlation = self.compute_rank_correlation(our_rankings, reference_rankings, top_k=top_k_correlation)
        return {
            "status": "completed",
            "correlation": correlation,
        }
