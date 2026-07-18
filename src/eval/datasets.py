import os
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader


DATASET_NAMES = {
    "hotpotqa": "hotpotqa",
    "scifact": "scifact",
    "fiqa": "fiqa-2018",
}


def download_dataset(dataset_key, data_dir="./data"):
    dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_dir, dataset_name)
    if os.path.exists(out_dir):
        return out_dir
    util.download_and_unzip(url, data_dir)
    return out_dir


def load_dataset(dataset_key, data_dir="./data"):
    dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
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
