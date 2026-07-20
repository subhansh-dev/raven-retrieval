import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ColbertEncoder(nn.Module):
    """ColBERT-style late interaction encoder.

    Produces per-token embeddings with an optional projection head.
    Supports:
    - Pretrained BERT backbone (default)
    - Random or Xavier-initialized projection
    - Fine-tuning on retrieval triples (see ColbertContrastiveEncoder)
    - Loading trained checkpoints via load_checkpoint()
    - Batched document encoding with attention-mask trimming
      (encode_documents) — the memory-safe way to encode a corpus
    """

    def __init__(self, model_name="bert-base-uncased", projection_dim=128,
                 projection_init="xavier", use_flash_attention=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {}
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        hidden_size = self.encoder.config.hidden_size

        # Projection head
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.projection_dim = projection_dim

        # Initialize projection weights
        if projection_init == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        # else: default random init

        # Learnable temperature for scoring
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    # ── Device helpers ───────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    # ── Encoding ─────────────────────────────────────────────────────

    def forward(self, texts, max_length=256, return_mask=False):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        token_embeddings = outputs.last_hidden_state
        projected = self.projection(token_embeddings)
        projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
        if return_mask:
            return projected, encoded["attention_mask"]
        return projected

    def encode_query(self, text, max_length=64):
        """Encode a single query. Returns (1, n_tokens, dim) — no padding
        for single inputs (tokenizer pads to longest in batch = itself)."""
        return self.forward([text], max_length=max_length, return_mask=False)

    def encode_document(self, text, max_length=256):
        """Encode a single document. Returns (1, n_tokens, dim), no padding."""
        return self.forward([text], max_length=max_length, return_mask=False)

    def encode_documents(self, texts, max_length=256, batch_size=32, show_progress=False):
        """Memory-safe batched corpus encoding with padding REMOVED.

        The old pattern (one forward pass for the whole corpus, or one text
        at a time) either OOMs or takes forever. This batches properly AND
        trims each document to its real token count using the attention
        mask — padding tokens never enter storage, MaxSim scoring, or
        pooling.

        Returns: list of numpy arrays, each (n_real_tokens_i, projection_dim).
        """
        all_embeddings = []
        rng = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(list(rng), desc="Encoding documents")
            except ImportError:
                pass
        with torch.no_grad():
            for i in rng:
                batch = texts[i:i + batch_size]
                projected, mask = self.forward(batch, max_length=max_length, return_mask=True)
                mask_np = mask.cpu().numpy()
                for j in range(projected.shape[0]):
                    n_real = int(mask_np[j].sum())
                    all_embeddings.append(projected[j, :n_real].cpu().numpy())
        return all_embeddings

    def encode_batch(self, texts, max_length=256, batch_size=32):
        """Legacy API: returns padded batch tensors (kept for compatibility)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.forward(batch, max_length=max_length)
            all_embeddings.append(embeddings)
        return all_embeddings

    def encode_pooled(self, texts, max_length=256, strategy="mean"):
        """Single-vector encoding via pooling over token embeddings.

        Strategies: 'mean', 'max', 'cls'. Mask-aware (pads excluded).
        Useful for clustering (GMM needs fixed-dim input).
        """
        projected, mask = self.forward(texts, max_length=max_length, return_mask=True)

        if strategy == "cls":
            return projected[:, 0, :]

        mask_expanded = mask.unsqueeze(-1).float()

        if strategy == "mean":
            pooled = (projected * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif strategy == "max":
            projected_masked = projected.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = projected_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

        return pooled

    # ── Checkpoint I/O ───────────────────────────────────────────────

    def save_checkpoint(self, path, extra=None):
        """Save model weights (+ optional training metadata) to a file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"model_state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path, strict=True):
        """Load weights saved by save_checkpoint / train_colbert.py.

        Handles both raw state_dicts and wrapped payloads with a
        "model_state_dict" key. Returns self for chaining.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded checkpoint from {path}")
        return self

    # ── Misc ─────────────────────────────────────────────────────────

    def get_temperature(self):
        """Get learned temperature parameter."""
        return torch.exp(self.log_tau)


class ColbertContrastiveEncoder(ColbertEncoder):
    """ColBERT encoder with contrastive training support.

    Adds loss functions for training on (query, positive, negative) triples
    or (query, positive) pairs with in-batch negatives.
    """

    def __init__(self, model_name="bert-base-uncased", projection_dim=128,
                 margin=0.5, temperature=0.05):
        super().__init__(model_name, projection_dim, projection_init="xavier")
        self.margin = margin
        self.temperature = temperature

    @staticmethod
    def _maxsim_torch(query_emb, doc_emb):
        """MaxSim score between two embedding matrices (torch, differentiable)."""
        # Handle batched input (batch, tokens, dim) -> squeeze to 2D
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        if doc_emb.dim() == 3:
            doc_emb = doc_emb.squeeze(0)
        q_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        d_norm = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)
        sim = torch.mm(q_norm, d_norm.t())
        return sim.max(dim=1).values.sum()

    def contrastive_loss(self, query_emb, pos_emb, neg_emb):
        """Pairwise contrastive loss: -log(sigmoid(maxsim(q,pos) - maxsim(q,neg)))."""
        pos_sim = self._maxsim_torch(query_emb, pos_emb)
        neg_sim = self._maxsim_torch(query_emb, neg_emb)
        loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8)
        return loss

    def in_batch_negatives_loss(self, query_emb, doc_emb):
        """InfoNCE loss with in-batch negatives.

        For a batch of (query, positive_doc) pairs, other docs in the batch
        serve as negatives. Uses the learned temperature (log_tau) so the
        sharpness of the distribution is trainable.
        """
        batch_size = len(query_emb)
        scores = torch.zeros(batch_size, batch_size, device=self.device)
        for i in range(batch_size):
            q_i = query_emb[i]
            if q_i.dim() == 1:
                q_i = q_i.unsqueeze(0)
            for j in range(batch_size):
                d_j = doc_emb[j]
                if d_j.dim() == 1:
                    d_j = d_j.unsqueeze(0)
                scores[i, j] = self._maxsim_torch(q_i, d_j)

        labels = torch.arange(batch_size, device=self.device)
        # Use learned temperature (falls back to self.temperature scale via exp(log_tau))
        temperature = torch.exp(self.log_tau).clamp(min=1e-3)
        loss = torch.nn.functional.cross_entropy(scores / temperature, labels)
        return loss
