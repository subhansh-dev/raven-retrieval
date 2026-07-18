"""HyDE: Hypothetical Document Embeddings (Gao et al., 2022).

Instead of embedding the query directly, generate a hypothetical answer
to the query, then embed THAT for retrieval. The intuition: a hypothetical
answer is semantically closer to actual documents than the raw question.

Pipeline: Query → LLM generates hypothetical doc → Embed hypothetical doc → Retrieve

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
           (ACL 2023)
"""

import numpy as np
import hashlib
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HyDERetriever:
    """HyDE retriever using a generative model to create hypothetical documents.

    The generated hypothetical document is then embedded and used for dense retrieval.
    This bridges the semantic gap between queries and documents.
    """

    def __init__(self, embedding_model_name="all-MiniLM-L6-v2",
                 generator_model_name=None, chunk_size=200, chunk_overlap=50):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None

        # Generator model for hypothetical documents
        self._generator = None
        self._generator_tokenizer = None
        self.generator_model_name = generator_model_name

    def _load_generator(self):
        """Load the generative model for hypothetical document creation."""
        if self._generator is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_name = self.generator_model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logger.info(f"Loading HyDE generator: {model_name}")

            self._generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._generator = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            if not torch.cuda.is_available():
                self._generator = self._generator.to("cpu")

            self._generator.eval()
            logger.info("HyDE generator loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load generator model: {e}")
            self._generator = "failed"

    def generate_hypothetical_document(self, query, max_new_tokens=200):
        """Generate a hypothetical document that would answer the query.

        If no generator model is available, uses a template-based approach.
        """
        if self._generator == "failed" or self._generator is None:
            # Template-based fallback
            return self._template_hypothetical(query)

        import torch

        prompt = f"""Write a detailed passage that directly answers the following question.
The passage should read like an excerpt from a scientific article or encyclopedia.

Question: {query}

Passage:"""

        inputs = self._generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(self._generator.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._generator_tokenizer.eos_token_id,
            )

        generated = self._generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part (after the prompt)
        if "Passage:" in generated:
            generated = generated.split("Passage:")[-1].strip()

        return generated

    def _template_hypothetical(self, query):
        """Template-based hypothetical document generation (no LLM needed).

        Creates a pseudo-document by expanding the query with contextual framing.
        """
        query_clean = query.strip().rstrip("?").rstrip(".")

        templates = [
            f"Based on research and analysis, {query_clean}. "
            f"This topic involves several key aspects that are well-documented in the literature. "
            f"Studies have shown that {query_clean.lower()} is an important area of investigation. "
            f"The evidence suggests multiple factors contribute to this phenomenon.",

            f"The answer to {query_clean} involves understanding the underlying mechanisms. "
            f"Research has demonstrated that this is a complex topic with multiple contributing factors. "
            f"Key findings include the relationship between various variables and outcomes. "
            f"This area continues to be actively studied.",

            f"{query_clean}. "
            f"This is a well-studied topic in the scientific literature. "
            f"Multiple studies have investigated the various aspects and contributing factors. "
            f"The consensus view, based on available evidence, provides important insights.",
        ]

        # Deterministic selection based on query hash
        idx = int(hashlib.md5(query.encode()).hexdigest(), 16) % len(templates)
        return templates[idx]

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
        """Index the corpus for dense retrieval."""
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
        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = np.array(embeddings)

        # L2 normalize
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / norms

        return self

    def retrieve(self, query, top_k=10, use_hyde=True):
        """Retrieve using HyDE or direct query embedding.

        Args:
            query: The search query
            top_k: Number of results to return
            use_hyde: If True, generate hypothetical doc first; if False, embed query directly
        """
        if use_hyde:
            hypothetical = self.generate_hypothetical_document(query)
            # Embed the hypothetical document
            query_embedding = self.embedding_model.encode([hypothetical])
        else:
            query_embedding = self.embedding_model.encode([query])

        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / query_norm

        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()

        # Aggregate scores by document ID (take max chunk score)
        doc_scores = {}
        for idx in range(len(scores)):
            doc_id = self.corpus_ids[idx].split("::")[0]
            if doc_id not in doc_scores or scores[idx] > doc_scores[doc_id]:
                doc_scores[doc_id] = float(scores[idx])

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def retrieve_batch(self, queries, top_k=10, use_hyde=True):
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k=top_k, use_hyde=use_hyde)
            all_results.append(results)
        return all_results
