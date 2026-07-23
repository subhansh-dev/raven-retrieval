"""Late Chunking (Jina AI, 2024).

Standard chunking: split text → embed each chunk independently.
Late chunking: embed entire document → split the embedding sequence.

The key insight: when you embed a chunk independently, the [CLS] token
has no awareness of the surrounding context. Late chunking preserves
cross-chunk context because the transformer sees the full document.

Reference: Günther et al., "Late Chunking: Contextual Chunk Embeddings
           Using Long-Context Embedding Models" (Jina AI, 2024)
           https://arxiv.org/abs/2409.04701
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class LateChunkingEncoder:
    """Encode full documents then chunk the token embeddings.

    Instead of chunking text THEN encoding, we:
    1. Encode the full document (up to max_length tokens)
    2. Split the resulting token embeddings by token position
    3. Pool each chunk's token embeddings

    This preserves cross-chunk context that would otherwise be lost.
    """

    def __init__(self, model_name="bert-base-uncased", chunk_size_tokens=128,
                 max_doc_length=2048, pooling="mean", model=None, tokenizer=None):
        from transformers import AutoTokenizer, AutoModel

        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer  # dependency injection for tests
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        self.chunk_size = chunk_size_tokens
        self.max_doc_length = max_doc_length
        self.pooling = pooling

    def _pool_chunk(self, token_embeddings, start, end):
        """Pool token embeddings for a chunk range."""
        chunk_embs = token_embeddings[start:end]

        if self.pooling == "mean":
            return chunk_embs.mean(dim=0).cpu().numpy()
        elif self.pooling == "max":
            return chunk_embs.max(dim=0).values.cpu().numpy()
        elif self.pooling == "cls":
            return chunk_embs[0].cpu().numpy()
        else:
            return chunk_embs.mean(dim=0).cpu().numpy()

    def encode_document_late_chunked(self, text):
        """Encode a document and return late-chunked embeddings.

        Returns list of chunk embeddings, each preserving cross-chunk context.
        """
        # Tokenize full document
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_doc_length,
            truncation=True,
            padding=False,
        )

        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Get full document token embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden)

        # Remove [CLS] and [SEP] tokens, but guard against very short documents
        # where removing both special tokens would leave nothing
        content_len = token_embeddings.shape[0] - 2
        if content_len <= 0:
            # Document was so short it only had [CLS] + [SEP] (or just [CLS])
            # Use the full sequence including special tokens as a single chunk
            logger.warning("Document too short for late chunking, using full token sequence")
            token_embeddings = outputs.last_hidden_state.squeeze(0)
            content_len = token_embeddings.shape[0]
        else:
            token_embeddings = token_embeddings[1:-1]  # (content_len, hidden)

        # Split into chunks by token position
        num_tokens = token_embeddings.shape[0]
        chunk_embeddings = []

        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_emb = self._pool_chunk(token_embeddings, start, end)
            chunk_embeddings.append(chunk_emb)

        return chunk_embeddings

    def encode_corpus(self, corpus, doc_ids=None):
        """Encode entire corpus with late chunking.

        Returns:
            all_chunk_embeddings: list of chunk embeddings
            all_chunk_ids: list of chunk IDs (doc_id::chunk::idx)
        """
        all_embeddings = []
        all_ids = []

        if isinstance(corpus, dict):
            corpus_iter = [(did, d) for did, d in corpus.items()]
        else:
            corpus_iter = [(doc["_id"], doc) for doc in corpus]

        for doc_id, doc in corpus_iter:
            full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()

            try:
                chunk_embs = self.encode_document_late_chunked(full_text)
                for i, emb in enumerate(chunk_embs):
                    all_embeddings.append(emb)
                    all_ids.append(f"{doc_id}::chunk::{i}")
            except Exception as e:
                logger.warning(f"Failed to encode {doc_id}: {e}")
                continue

        return all_embeddings, all_ids


class LateChunkingRetriever:
    """Retrieval system using late-chunked document embeddings."""

    def __init__(self, model_name="bert-base-uncased", chunk_size_tokens=128,
                 max_doc_length=2048):
        self.encoder = LateChunkingEncoder(
            model_name=model_name,
            chunk_size_tokens=chunk_size_tokens,
            max_doc_length=max_doc_length,
        )
        self.corpus_ids = []
        self.corpus_embeddings = None

    def index(self, corpus):
        """Index corpus with late chunking."""
        logger.info("Encoding corpus with late chunking...")
        all_embeddings, all_ids = self.encoder.encode_corpus(corpus)

        self.corpus_ids = all_ids
        self.corpus_embeddings = np.array(all_embeddings)

        # L2 normalize
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / norms

        return self

    def retrieve(self, query, top_k=10):
        """Retrieve using late-chunked embeddings."""
        # Encode query (single chunk, no late chunking needed)
        query_tokens = self.encoder.tokenizer(
            query, return_tensors="pt", max_length=128, truncation=True
        )
        device = next(self.encoder.model.parameters()).device
        query_tokens = {k: v.to(device) for k, v in query_tokens.items()}

        with torch.no_grad():
            query_output = self.encoder.model(**query_tokens)
            query_emb = query_output.last_hidden_state.squeeze(0)
            # Pool: skip [CLS] and [SEP] only if there are content tokens left
            if query_emb.shape[0] > 2:
                query_emb = query_emb[1:-1].mean(dim=0).cpu().numpy()
            else:
                # Very short query — use CLS token as fallback
                query_emb = query_emb[0].cpu().numpy()

        query_emb = query_emb / np.linalg.norm(query_emb)

        # Cosine similarity
        scores = np.dot(self.corpus_embeddings, query_emb)

        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
