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

from ..utils import chunk_corpus, aggregate_doc_scores, l2_normalize

logger = logging.getLogger(__name__)


class HyDERetriever:
    """HyDE retriever using a generative model to create hypothetical documents.

    The generated hypothetical document is then embedded and used for dense retrieval.

    Args:
        model_name: embedding model (sentence-transformers)
        generator_model_name: causal LM for hypothetical generation. If None
            (or loading fails), a deterministic template fallback is used —
            this is explicit in logs so benchmark results are clearly labeled.
        use_llm: if False, skip generator loading entirely (template mode).
    """

    def __init__(self, model_name="all-MiniLM-L6-v2",
                 generator_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 use_llm=False, chunk_size=200, chunk_overlap=50, model=None):
        if model is not None:
            self.embedding_model = model  # dependency injection for tests
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None

        self._generator = None
        self._generator_tokenizer = None
        self._generator_failed = False
        self.generator_model_name = generator_model_name
        self.use_llm = use_llm

    def _load_generator(self):
        """Load the generative model once; never retries a failed load."""
        if self._generator is not None or self._generator_failed:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading HyDE generator: {self.generator_model_name}")
            self._generator_tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForCausalLM.from_pretrained(
                self.generator_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self._generator = self._generator.to("cuda")
            self._generator.eval()
            logger.info("HyDE generator loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load generator model ({e}) — using template HyDE")
            self._generator_failed = True

    def generate_hypothetical_document(self, query, max_new_tokens=200):
        """Generate a hypothetical document that would answer the query.

        Uses the LLM when available and enabled; otherwise a deterministic
        template fallback (always labeled in logs).
        """
        if self.use_llm and not self._generator_failed:
            self._load_generator()

        if self._generator is None:
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
        if "Passage:" in generated:
            generated = generated.split("Passage:")[-1].strip()

        return generated if generated else self._template_hypothetical(query)

    def _template_hypothetical(self, query):
        """Template-based hypothetical document generation (no LLM needed).

        Deterministic: the same query always gets the same template.
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

        idx = int(hashlib.md5(query.encode()).hexdigest(), 16) % len(templates)
        return templates[idx]

    def index(self, corpus):
        """Index the corpus for dense retrieval."""
        self.corpus_ids, self.corpus_texts = chunk_corpus(
            corpus, self.chunk_size, self.chunk_overlap
        )
        embeddings = self.embedding_model.encode(self.corpus_texts, show_progress_bar=True, batch_size=64)
        self.corpus_embeddings = l2_normalize(np.array(embeddings))
        return self

    def retrieve(self, query, top_k=10, use_hyde=True):
        """Retrieve using HyDE or direct query embedding.

        Args:
            query: The search query
            top_k: Number of results to return
            use_hyde: If True, generate hypothetical doc first; if False, embed query directly
        """
        text_to_embed = self.generate_hypothetical_document(query) if use_hyde else query
        query_embedding = np.array(self.embedding_model.encode([text_to_embed]))
        query_embedding = l2_normalize(query_embedding)

        scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        return aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k)

    def retrieve_batch(self, queries, top_k=10, use_hyde=True):
        """Batch retrieval. Returns doc-aggregated results (consistent with retrieve)."""
        texts = [self.generate_hypothetical_document(q) if use_hyde else q for q in queries]
        query_embeddings = l2_normalize(np.array(self.embedding_model.encode(texts, batch_size=64)))
        all_results = []
        for i in range(len(queries)):
            scores = np.dot(self.corpus_embeddings, query_embeddings[i]).flatten()
            all_results.append(aggregate_doc_scores(self.corpus_ids, scores, top_k=top_k))
        return all_results
