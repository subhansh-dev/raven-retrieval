import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel


class ColbertEncoder(nn.Module):
    """ColBERT-style late interaction encoder.

    Produces per-token embeddings with an optional projection head.
    Supports:
    - Pretrained BERT backbone (default)
    - Random or Xavier-initialized projection
    - Optional fine-tuning on retrieval triples
    - Mean/max pooling for single-vector fallback
    - Flash Attention 2 when available
    """

    def __init__(self, model_name="bert-base-uncased", projection_dim=128,
                 projection_init="xavier", use_flash_attention=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with optional Flash Attention 2
        model_kwargs = {}
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass  # Fall back to standard attention

        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        hidden_size = self.encoder.config.hidden_size

        # Projection head
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.projection_dim = projection_dim

        # Initialize projection weights
        if projection_init == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        # else: default random init (as in original)

        # Learnable temperature for scoring
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, texts, max_length=256, return_mask=False):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        token_embeddings = outputs.last_hidden_state
        projected = self.projection(token_embeddings)
        projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
        if return_mask:
            return projected, encoded["attention_mask"]
        return projected

    def encode_query(self, text, max_length=64):
        return self.forward([text], max_length=max_length, return_mask=False)

    def encode_document(self, text, max_length=256):
        return self.forward([text], max_length=max_length, return_mask=False)

    def encode_batch(self, texts, max_length=256, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.forward(batch, max_length=max_length)
            all_embeddings.append(embeddings)
        return all_embeddings

    def encode_pooled(self, texts, max_length=256, strategy="mean"):
        """Single-vector encoding via pooling over token embeddings.

        Strategies: 'mean', 'max', 'cls'
        Useful for clustering (GMM needs fixed-dim input).
        """
        projected, mask = self.forward(texts, max_length=max_length, return_mask=True)

        if strategy == "cls":
            return projected[:, 0, :]  # CLS token

        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()

        if strategy == "mean":
            pooled = (projected * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif strategy == "max":
            projected_masked = projected.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = projected_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

        return pooled

    def get_temperature(self):
        """Get learned temperature parameter."""
        return torch.exp(self.log_tau)


class ColbertContrastiveEncoder(ColbertEncoder):
    """ColBERT encoder with contrastive training support.

    Adds a contrastive loss function for training on (query, positive, negative) triples.
    This is what's needed to actually train the projection head instead of using random init.
    """

    def __init__(self, model_name="bert-base-uncased", projection_dim=128,
                 margin=0.5, temperature=0.05):
        super().__init__(model_name, projection_dim, projection_init="xavier")
        self.margin = margin
        self.temperature = temperature

    def contrastive_loss(self, query_emb, pos_emb, neg_emb):
        """Compute ColBERT-style contrastive loss.

        Loss = -log(sigmoid(maxsim(q, pos) - maxsim(q, neg)))
        """
        # MaxSim scores
        q_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        p_norm = torch.nn.functional.normalize(pos_emb, p=2, dim=-1)
        n_norm = torch.nn.functional.normalize(neg_emb, p=2, dim=-1)

        pos_sim = torch.mm(q_norm, p_norm.t()).max(dim=1).values.sum()
        neg_sim = torch.mm(q_norm, n_norm.t()).max(dim=1).values.sum()

        # Contrastive loss
        loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8)
        return loss

    def in_batch_negatives_loss(self, query_emb, doc_emb):
        """InfoNCE-style loss with in-batch negatives.

        For a batch of (query, positive_doc) pairs,
        other docs in the batch serve as negatives.
        """
        batch_size = len(query_emb)

        # Compute pairwise MaxSim scores
        scores = torch.zeros(batch_size, batch_size, device=query_emb.device)
        for i in range(batch_size):
            for j in range(batch_size):
                q_i = query_emb[i]
                if q_i.dim() == 1:
                    q_i = q_i.unsqueeze(0)
                d_j = doc_emb[j]
                if d_j.dim() == 1:
                    d_j = d_j.unsqueeze(0)
                q_norm = torch.nn.functional.normalize(q_i, p=2, dim=-1)
                d_norm = torch.nn.functional.normalize(d_j, p=2, dim=-1)
                sim = torch.mm(q_norm, d_norm.t())
                scores[i, j] = sim.max(dim=1).values.sum()

        # InfoNCE loss
        labels = torch.arange(batch_size, device=query_emb.device)
        loss = torch.nn.functional.cross_entropy(scores / self.temperature, labels)
        return loss
