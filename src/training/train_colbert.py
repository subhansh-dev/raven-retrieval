"""Training script for ColBERT Contrastive Encoder.

Trains the projection head on retrieval triples (query, positive, negative)
using contrastive loss or InfoNCE with in-batch negatives.

Usage:
    python -m src.training.train_colbert --triples data/triples.jsonl --epochs 3
    python -m src.training.train_colbert --beir-dataset scifact --epochs 5

The script can also generate synthetic training data from BEIR datasets
using the existing qrels (positive) and non-relevant documents (negative).
"""

import os
import sys
import json
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_triples_from_beir(corpus, queries, qrels, max_triples=10000, seed=42):
    """Generate (query, positive_doc, negative_doc) triples from BEIR data.

    Positive: documents with relevance judgment > 0
    Negative: random non-relevant documents (hard negatives)
    """
    rng = np.random.RandomState(seed)
    triples = []

    # Build corpus lookup
    all_doc_ids = list(corpus.keys())

    for qid, relevant_docs in qrels.items():
        if qid not in queries:
            continue

        query_text = queries[qid]
        pos_doc_ids = [did for did, score in relevant_docs.items() if score > 0]

        if not pos_doc_ids:
            continue

        # Sample negatives
        neg_candidates = [did for did in all_doc_ids if did not in relevant_docs]
        if not neg_candidates:
            continue

        for pos_id in pos_doc_ids[:3]:  # Limit positives per query
            neg_id = rng.choice(neg_candidates)

            pos_doc = corpus[pos_id]
            neg_doc = corpus[neg_id]

            pos_text = (pos_doc.get("title", "") + " " + pos_doc.get("text", "")).strip()
            neg_text = (neg_doc.get("title", "") + " " + neg_doc.get("text", "")).strip()

            triples.append({
                "query": query_text,
                "positive": pos_text[:1000],
                "negative": neg_text[:1000],
            })

            if len(triples) >= max_triples:
                break

        if len(triples) >= max_triples:
            break

    rng.shuffle(triples)
    return triples


def train_colbert_encoder(
    encoder,
    triples,
    epochs=3,
    batch_size=8,
    learning_rate=3e-5,
    warmup_steps=100,
    max_grad_norm=1.0,
    output_dir="./checkpoints",
    eval_every=500,
):
    """Train ColBERT encoder with contrastive loss.

    Args:
        encoder: ColbertContrastiveEncoder instance
        triples: list of {"query": ..., "positive": ..., "negative": ...}
        epochs: number of training epochs
        batch_size: training batch size
        learning_rate: optimizer learning rate
        warmup_steps: linear warmup steps
        max_grad_norm: gradient clipping
        output_dir: where to save checkpoints
        eval_every: evaluate every N steps
    """
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.train()

    optimizer = AdamW(encoder.parameters(), lr=learning_rate, weight_decay=0.01)

    # Linear warmup then linear decay
    total_steps = len(triples) * epochs // batch_size
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs(output_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        np.random.shuffle(triples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            if len(batch) < 2:
                continue

            queries = [t["query"] for t in batch]
            positives = [t["positive"] for t in batch]
            negatives = [t["negative"] for t in batch]

            # Encode
            query_embs = encoder.forward(queries, max_length=64)
            pos_embs = encoder.forward(positives, max_length=256)
            neg_embs = encoder.forward(negatives, max_length=256)

            # Compute loss
            if len(batch) >= 4:
                # Use InfoNCE with in-batch negatives
                loss = encoder.in_batch_negatives_loss(
                    [query_embs[j] for j in range(len(batch))],
                    [pos_embs[j] for j in range(len(batch))],
                )
            else:
                # Use pairwise contrastive loss
                loss = torch.tensor(0.0, device=device)
                for j in range(len(batch)):
                    loss += encoder.contrastive_loss(
                        query_embs[j].unsqueeze(0),
                        pos_embs[j].unsqueeze(0),
                        neg_embs[j].unsqueeze(0),
                    )
                loss /= len(batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % eval_every == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")

                # Save checkpoint if best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        "epoch": epoch,
                        "step": global_step,
                        "model_state_dict": encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                    }, os.path.join(output_dir, "best_model.pt"))
                    logger.info(f"Saved best model (loss={best_loss:.4f})")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_epoch_loss:.4f}")

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": encoder.state_dict(),
        "loss": epoch_loss / max(n_batches, 1),
    }, os.path.join(output_dir, "final_model.pt"))

    logger.info(f"Training complete. Models saved to {output_dir}")
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Train ColBERT Contrastive Encoder")
    parser.add_argument("--beir-dataset", default="scifact", help="BEIR dataset for training data")
    parser.add_argument("--triples", default=None, help="Path to pre-generated triples JSONL")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-triples", type=int, default=10000)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--encoder-model", default="bert-base-uncased")
    parser.add_argument("--projection-dim", type=int, default=128)
    args = parser.parse_args()

    # Import here to avoid torch dependency at module level
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.encoder.colbert_encoder import ColbertContrastiveEncoder

    encoder = ColbertContrastiveEncoder(
        model_name=args.encoder_model,
        projection_dim=args.projection_dim,
    )

    if args.triples:
        # Load pre-generated triples
        triples = []
        with open(args.triples) as f:
            for line in f:
                triples.append(json.loads(line))
        logger.info(f"Loaded {len(triples)} triples from {args.triples}")
    else:
        # Generate from BEIR
        from src.eval.datasets import load_dataset
        logger.info(f"Generating triples from {args.beir_dataset}...")
        corpus, queries, qrels = load_dataset(args.beir_dataset)
        triples = generate_triples_from_beir(
            corpus, queries, qrels,
            max_triples=args.max_triples,
        )
        logger.info(f"Generated {len(triples)} triples")

        # Save triples for reproducibility
        os.makedirs(args.output_dir, exist_ok=True)
        triples_path = os.path.join(args.output_dir, "triples.jsonl")
        with open(triples_path, "w") as f:
            for t in triples:
                f.write(json.dumps(t) + "\n")
        logger.info(f"Triples saved to {triples_path}")

    # Train
    trained_encoder = train_colbert_encoder(
        encoder,
        triples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
