import os
import logging

logger = logging.getLogger(__name__)


DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "scifact": "scifact",
    "fiqa": "fiqa-2018",
}


def download_dataset(dataset_key, data_dir="./data"):
    from beir import util
    dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_dir, dataset_name)
    if os.path.exists(out_dir):
        return out_dir
    util.download_and_unzip(url, data_dir)
    return out_dir


def load_dataset(dataset_key, data_dir="./data"):
    from beir.datasets.data_loader import GenericDataLoader
    out_dir = download_dataset(dataset_key, data_dir)
    corpus, queries, qrels = GenericDataLoader(out_dir).load(split="test")
    return corpus, queries, qrels


def get_corpus_texts(corpus):
    texts = []
    ids = []
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        full_text = (title + " " + text).strip()
        texts.append(full_text)
        ids.append(doc_id)
    return ids, texts


def get_query_list(queries, qrels=None):
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    return query_ids, query_texts


def subsample_queries(queries, qrels, n_samples, seed=42):
    import random
    rng = random.Random(seed)
    query_ids = list(queries.keys())
    if n_samples >= len(query_ids):
        return queries, qrels
    sampled_ids = rng.sample(query_ids, n_samples)
    sampled_queries = {qid: queries[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels[qid] for qid in sampled_ids if qid in qrels}
    return sampled_queries, sampled_qrels


def subsample_corpus(corpus, qrels, max_docs, seed=42):
    """Subsample a corpus to at most max_docs documents.

    CRITICAL correctness property: every document referenced by the qrels
    is ALWAYS kept. Dropping judged documents would silently corrupt every
    metric (relevant docs would become unretrievable). The remaining budget
    is filled with a seeded random sample of the rest.

    This is what makes HotpotQA (5.2M docs) benchmarkable on a 4-8GB laptop.
    """
    if max_docs is None or len(corpus) <= max_docs:
        return corpus

    import random
    rng = random.Random(seed)

    judged_ids = set()
    for rels in qrels.values():
        for doc_id, rel in rels.items():
            if rel > 0 and doc_id in corpus:
                judged_ids.add(doc_id)

    remaining_budget = max_docs - len(judged_ids)
    if remaining_budget < 0:
        logger.warning(
            f"max_docs={max_docs} is smaller than the number of judged docs "
            f"({len(judged_ids)}) — keeping all judged docs anyway (metrics stay valid)."
        )
        remaining_budget = 0

    other_ids = [d for d in corpus.keys() if d not in judged_ids]
    sampled_others = rng.sample(other_ids, min(remaining_budget, len(other_ids)))

    keep = judged_ids | set(sampled_others)
    subsampled = {d: corpus[d] for d in keep}
    logger.info(
        f"Corpus subsampled: {len(corpus)} -> {len(subsampled)} docs "
        f"({len(judged_ids)} judged docs preserved)"
    )
    return subsampled
