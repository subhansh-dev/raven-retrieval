"""Agentic RAG: Query Decomposition & Reflection.

Implements two key agentic patterns:
1. Query Decomposition: break complex queries into sub-queries
2. Reflection: evaluate if retrieved context is sufficient, re-retrieve if not

Reference: Survey of Agentic RAG (2025) - https://arxiv.org/abs/2501.09136
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decompose complex queries into simpler sub-queries.

    For multi-hop questions like "What country is the capital of the
    company that made the iPhone located in?", decomposes into:
    1. "What company made the iPhone?"
    2. "What is the capital of Apple's country?"

    Uses template-based decomposition (no LLM needed) with optional
    LLM-based decomposition for more complex queries.
    """

    def __init__(self, llm_model_name=None):
        self._llm = None
        self._tokenizer = None
        self.llm_model_name = llm_model_name

    def _load_llm(self):
        if self._llm is not None:
            return
        if self.llm_model_name is None:
            self._llm = "unavailable"
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if not torch.cuda.is_available():
                self._llm = self._llm.to("cpu")
            self._llm.eval()
        except Exception as e:
            logger.warning(f"Failed to load LLM for decomposition: {e}")
            self._llm = "unavailable"

    def _template_decompose(self, query):
        """Template-based query decomposition.

        Uses heuristics to detect multi-hop queries and decompose them.
        """
        query_lower = query.lower().strip()

        # Detect conjunction-based multi-hop
        conjunctions = [" and ", " but also ", " as well as ", " in addition to "]
        for conj in conjunctions:
            if conj in query_lower:
                parts = query.split(conj)
                if len(parts) == 2:
                    return [p.strip() for p in parts if p.strip()]

        # Detect comparison queries
        comparison_patterns = ["compare", "difference between", "vs", "versus", " or "]
        for pattern in comparison_patterns:
            if pattern in query_lower:
                # Keep as single query — comparisons need full context
                return [query]

        # Detect "who is the X of Y" chains
        if " who " in query_lower or " what " in query_lower:
            # Try to split on relative clauses
            if " that " in query_lower or " which " in query_lower:
                for sep in [" that ", " which "]:
                    if sep in query_lower:
                        parts = query.split(sep, 1)
                        if len(parts) == 2:
                            return [parts[0].strip(), parts[1].strip()]

        # Default: single query
        return [query]

    def decompose(self, query, use_llm=False):
        """Decompose a query into sub-queries.

        Args:
            query: The original query
            use_llm: If True, try LLM-based decomposition; if False or unavailable, use templates

        Returns: list of sub-queries (at minimum, [query])
        """
        if use_llm:
            self._load_llm()
            if self._llm not in (None, "unavailable"):
                return self._llm_decompose(query)

        return self._template_decompose(query)

    def _llm_decompose(self, query):
        """LLM-based query decomposition."""
        import torch

        prompt = f"""Decompose the following question into simpler sub-questions that can be answered independently.
Return each sub-question on a new line. If the question is already simple, return it as-is.

Question: {query}

Sub-questions:"""

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(self._llm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Sub-questions:" in generated:
            generated = generated.split("Sub-questions:")[-1].strip()

        sub_queries = [q.strip() for q in generated.split("\n") if q.strip() and len(q.strip()) > 5]
        return sub_queries if sub_queries else [query]


class ReflectionRetriever:
    """Reflection-based retrieval: evaluate if context is sufficient, re-retrieve if not.

    Pattern:
    1. Initial retrieval
    2. Evaluate: "Is this context sufficient to answer the question?"
    3. If not, reformulate query and re-retrieve
    4. Repeat up to max_iterations

    Reference: Self-RAG (Asai et al., 2023)

    Args:
        base_retriever: any retriever with .retrieve(query, top_k)
        text_lookup: callable(doc_id) -> document text. REQUIRED for real
            sufficiency evaluation — the old version evaluated keyword
            coverage over doc ID STRINGS (which never contain query
            keywords), so reflection triggered (or didn't) essentially
            at random. If None, falls back to retriever.corpus_texts
            when available.
        max_iterations: max retrieve-evaluate-reformulate cycles
        top_k: results per retrieval round
    """

    def __init__(self, base_retriever, text_lookup=None, max_iterations=2, top_k=10):
        self.base_retriever = base_retriever
        self.max_iterations = max_iterations
        self.top_k = top_k
        if text_lookup is None:
            text_lookup = self._build_default_lookup(base_retriever)
        self.text_lookup = text_lookup

    @staticmethod
    def _build_default_lookup(base_retriever):
        """Try to build a text lookup from the retriever's own index."""
        corpus_ids = getattr(base_retriever, "corpus_ids", None)
        corpus_texts = getattr(base_retriever, "corpus_texts", None)
        if corpus_ids and corpus_texts:
            chunk_map = {}
            for cid, text in zip(corpus_ids, corpus_texts):
                doc_id = cid.split("::")[0]
                if doc_id not in chunk_map:
                    chunk_map[doc_id] = []
                chunk_map[doc_id].append(text)

            def lookup(doc_id):
                return " ".join(chunk_map.get(doc_id, []))
            return lookup
        return None

    def _evaluate_context(self, query, context_texts):
        """Evaluate if retrieved context is sufficient.

        Uses simple heuristics (keyword overlap) instead of LLM
        for speed. Can be upgraded to LLM-based evaluation.
        """
        query_tokens = set(query.lower().split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "of", "in", "to", "for", "with", "on", "at", "from", "by",
                      "about", "as", "into", "through", "during", "before", "after",
                      "what", "which", "who", "whom", "this", "that", "these", "those",
                      "and", "but", "or", "nor", "not", "so", "yet"}
        query_keywords = query_tokens - stop_words

        if not query_keywords:
            return True, 0.0

        # Check keyword coverage in context
        context_text = " ".join(context_texts).lower()
        covered = sum(1 for kw in query_keywords if kw in context_text)
        coverage = covered / len(query_keywords)

        # Threshold: at least 50% keyword coverage
        sufficient = coverage >= 0.5

        return sufficient, coverage

    def _reformulate_query(self, query, context_texts, iteration):
        """Reformulate query based on what's missing.

        Simple approach: add specificity or broaden based on iteration.
        """
        query_lower = query.lower()

        if iteration == 1:
            # First reformulation: try more specific version
            # Remove question words and add key terms from context
            reformulated = query
            for prefix in ["what is ", "what are ", "who is ", "who was ",
                           "how does ", "how do ", "why does ", "why do "]:
                if query_lower.startswith(prefix):
                    reformulated = query[len(prefix):]
                    break
            return reformulated
        else:
            # Second reformulation: try broader version
            words = query.split()
            if len(words) > 5:
                return " ".join(words[:len(words)//2])
            return query

    def retrieve(self, query, top_k=None):
        """Retrieve with reflection loop.

        Returns aggregated results from multiple retrieval iterations.
        """
        if top_k is None:
            top_k = self.top_k

        all_results = []
        seen_doc_ids = set()

        current_query = query

        for iteration in range(self.max_iterations):
            # Retrieve
            results = self.base_retriever.retrieve(current_query, top_k=top_k)

            # Add new results
            for doc_id, score in results:
                if doc_id not in seen_doc_ids:
                    all_results.append((doc_id, score))
                    seen_doc_ids.add(doc_id)

            # Evaluate context sufficiency over actual document TEXT
            if self.text_lookup is not None:
                context_texts = [self.text_lookup(doc_id) for doc_id, _ in results]
            else:
                logger.warning("ReflectionRetriever: no text_lookup available — "
                               "sufficiency evaluation disabled, returning first-pass results")
                break
            sufficient, coverage = self._evaluate_context(query, context_texts)

            if sufficient:
                break

            # Reformulate and retry
            current_query = self._reformulate_query(query, context_texts, iteration)
            logger.debug(f"Reflection iteration {iteration}: reformulated to '{current_query}'")

        # Sort by score and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]


class MultiHopRetriever:
    """Multi-hop retrieval with query decomposition.

    For complex queries that require information from multiple documents:
    1. Decompose query into sub-queries
    2. Retrieve for each sub-query
    3. Aggregate results with score fusion
    """

    def __init__(self, base_retriever, decomposer=None, top_k=10):
        self.base_retriever = base_retriever
        self.decomposer = decomposer or QueryDecomposer()
        self.top_k = top_k

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k

        sub_queries = self.decomposer.decompose(query)

        if len(sub_queries) <= 1:
            return self.base_retriever.retrieve(query, top_k=top_k)

        # Retrieve for each sub-query
        doc_scores = {}
        for sq in sub_queries:
            results = self.base_retriever.retrieve(sq, top_k=top_k)
            for doc_id, score in results:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                doc_scores[doc_id].append(score)

        # Aggregate: sum of scores across sub-queries
        aggregated = []
        for doc_id, scores in doc_scores.items():
            aggregated.append((doc_id, sum(scores)))

        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:top_k]
