import re


class TextChunker:

    def __init__(self, chunk_size=100, overlap=0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text, doc_id=None):
        tokens = text.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            chunk_id = f"{doc_id}::l0::{idx}" if doc_id else f"l0::{idx}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "token_count": end - start,
                "level": 0,
                "doc_id": doc_id,
            })
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks

    def chunk_corpus(self, corpus):
        all_chunks = []
        if isinstance(corpus, dict):
            corpus_iter = [(did, d) for did, d in corpus.items()]
        else:
            corpus_iter = [(doc["_id"], doc) for doc in corpus]
        for doc_id, doc in corpus_iter:
            full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            chunks = self.chunk(full_text, doc_id=doc_id)
            all_chunks.extend(chunks)
        return all_chunks
