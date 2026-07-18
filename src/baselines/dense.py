import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None

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
        embeddings = self.model.encode(all_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = np.array(embeddings)
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / norms
        return self

    def retrieve(self, query, top_k=10):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / query_norm
        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def retrieve_batch(self, queries, top_k=10):
        query_embeddings = self.model.encode(queries, show_progress_bar=True, batch_size=64)
        query_embeddings = np.array(query_embeddings)
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / query_norms
        all_results = []
        for i in range(len(queries)):
            scores = np.dot(self.corpus_embeddings, query_embeddings[i]).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self.corpus_ids[idx], float(scores[idx])))
            all_results.append(results)
        return all_results
