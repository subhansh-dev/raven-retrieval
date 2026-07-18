import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50, rrf_k=60):
        self.dense_model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None
        self.bm25 = None

    def chunk_text(self, text):
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def index(self, corpus):
        all_texts = []
        all_ids = []
        if isinstance(corpus, dict):
            corpus_iter = [(did, d) for did, d in corpus.items()]
        else:
            corpus_iter = [(doc["_id"], doc) for doc in corpus]
        for doc_id, doc in corpus_iter:
            full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk::{i}"
                all_texts.append(chunk)
                all_ids.append(chunk_id)
        self.corpus_ids = all_ids
        self.corpus_texts = all_texts
        tokenized_corpus = [text.lower().split() for text in all_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        embeddings = self.dense_model.encode(all_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = np.array(embeddings)
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / norms
        return self

    def reciprocal_rank_fusion(self, rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def retrieve(self, query, top_k=10):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_doc_scores = {}
        for idx in range(len(bm25_scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in bm25_doc_scores or bm25_scores[idx] > bm25_doc_scores[doc_id]:
                bm25_doc_scores[doc_id] = float(bm25_scores[idx])
        bm25_sorted = sorted(bm25_doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        bm25_ranking = [(did, s) for did, s in bm25_sorted]
        query_embedding = self.dense_model.encode([query])
        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / query_norm
        dense_scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        dense_doc_scores = {}
        for idx in range(len(dense_scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in dense_doc_scores or dense_scores[idx] > dense_doc_scores[doc_id]:
                dense_doc_scores[doc_id] = float(dense_scores[idx])
        dense_sorted = sorted(dense_doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        dense_ranking = [(did, s) for did, s in dense_sorted]
        fused = self.reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=self.rrf_k)
        return fused[:top_k]

    def retrieve_batch(self, queries, top_k=10):
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            all_results.append(results)
        return all_results
