import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Abstractive summarizer using seq2seq models (BART, T5, etc.).

    Compatible with transformers 4.x and 5.x by using AutoModelForSeq2SeqLM
    directly instead of the deprecated pipeline("text2text-generation") API.

    Falls back to extractive summarization if the model fails to load.
    """

    def __init__(self, model_name="facebook/bart-large-cnn", max_chunk_tokens=1024, max_summary_tokens=200,
                 device=None, fallback_to_extractive=True):
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.max_summary_tokens = max_summary_tokens
        self.fallback_to_extractive = fallback_to_extractive
        self._model = None
        self._tokenizer = None
        self._device = device
        self._use_extractive = False
        self._load_attempted = False  # prevents retrying a failed load forever

    def _load_model(self):
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            if self._device is not None:
                self._model = self._model.to(self._device)
            elif torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                self._model = self._model.to("cpu")

            self._model.eval()
            logger.info(f"Loaded summarizer model: {self.model_name}")

        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            self._model = None
            if self.fallback_to_extractive:
                logger.info("Falling back to extractive summarization")
                self._use_extractive = True
            else:
                raise

    def _extractive_summarize(self, text, max_sentences=3):
        """Simple extractive summarization using sentence scoring.

        Scores sentences by position and length — no model needed.
        """
        if not text or not text.strip():
            return ""

        # Split into sentences
        sentences = []
        for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            if not sentences:
                sentences = text.split(sep)
            else:
                new_sentences = []
                for s in sentences:
                    new_sentences.extend(s.split(sep))
                sentences = new_sentences

        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if not sentences:
            return text[:500]

        if len(sentences) <= max_sentences:
            return ". ".join(sentences)

        # Score: favor early sentences + longer ones (more informative)
        scored = []
        for i, sent in enumerate(sentences):
            position_score = 1.0 / (1.0 + i * 0.3)  # decay
            length_score = min(len(sent.split()) / 20.0, 1.0)
            score = 0.6 * position_score + 0.4 * length_score
            scored.append((i, score, sent))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = sorted(scored[:max_sentences], key=lambda x: x[0])
        return ". ".join(s[2] for s in selected)

    def _abstractive_summarize(self, text):
        """Generate abstractive summary using the seq2seq model."""
        import torch

        self._load_model()
        if self._use_extractive:
            return self._extractive_summarize(text)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_chunk_tokens,
            truncation=True,
            padding=True,
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self._model.generate(
                **inputs,
                max_length=self.max_summary_tokens,
                min_length=30,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def summarize(self, text):
        """Summarize a single text. Returns empty string for empty input."""
        if not text or not text.strip():
            return ""

        words = text.split()
        if len(words) > self.max_chunk_tokens * 4:
            words = words[:self.max_chunk_tokens * 4]
            text = " ".join(words)

        try:
            return self._abstractive_summarize(text)
        except Exception as e:
            logger.warning(f"Abstractive summarization failed: {e}")
            if self.fallback_to_extractive:
                return self._extractive_summarize(text)
            raise

    def summarize_cluster(self, texts):
        """Summarize a cluster of texts by concatenating and summarizing."""
        concatenated = " ".join(texts)
        words = concatenated.split()
        if len(words) > self.max_chunk_tokens * 4:
            concatenated = " ".join(words[:self.max_chunk_tokens * 4])
        return self.summarize(concatenated)

    def summarize_with_budget(self, texts, token_budget=None):
        """Summarize with explicit token budget."""
        if token_budget is None:
            token_budget = self.max_chunk_tokens * 4
        concatenated = " ".join(texts)
        words = concatenated.split()
        if len(words) > token_budget:
            concatenated = " ".join(words[:token_budget])
        return self.summarize(concatenated)

    def generate_node_id(self, text, level, cluster_id):
        """Generate deterministic node ID from content."""
        hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"l{level}::c{cluster_id}::{hash_val}"


class ExtractiveSummarizer:
    """Lightweight extractive summarizer — no model required.

    Uses TF-IDF sentence scoring for fast, memory-efficient summarization.
    Good as a RAPTOR summarizer for large corpora where LLM summarization
    is too expensive.
    """

    def __init__(self, max_sentences=5):
        self.max_sentences = max_sentences

    def _score_sentences(self, sentences):
        """Score sentences using TF-IDF-like approach."""
        if not sentences:
            return []

        # Build word frequency across all sentences
        word_freq = {}
        for sent in sentences:
            for word in sent.lower().split():
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Normalize
        max_freq = max(word_freq.values()) if word_freq else 1
        for w in word_freq:
            word_freq[w] /= max_freq

        # Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            words = sent.lower().split()
            if not words:
                scored.append((i, 0.0))
                continue
            tf_score = sum(word_freq.get(w, 0) for w in words) / len(words)
            position_score = 1.0 / (1.0 + i * 0.2)
            length_score = min(len(words) / 15.0, 1.0)
            score = 0.5 * tf_score + 0.3 * position_score + 0.2 * length_score
            scored.append((i, score))

        return scored

    def summarize(self, text):
        """Extract top sentences as summary."""
        if not text or not text.strip():
            return ""

        # Split into sentences
        sentences = []
        for sep in [". ", "! ", "? ", ".\n"]:
            if not sentences:
                sentences = text.split(sep)
            else:
                new_sents = []
                for s in sentences:
                    new_sents.extend(s.split(sep))
                sentences = new_sents

        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if not sentences:
            return text[:500]
        if len(sentences) <= self.max_sentences:
            return ". ".join(sentences)

        scored = self._score_sentences(sentences)
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = sorted(scored[:self.max_sentences], key=lambda x: x[0])
        return ". ".join(sentences[s[0]] for s in selected)

    def summarize_cluster(self, texts):
        concatenated = " ".join(texts)
        return self.summarize(concatenated)

    def generate_node_id(self, text, level, cluster_id):
        hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"l{level}::c{cluster_id}::{hash_val}"
