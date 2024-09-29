import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import tqdm
from beir.retrieval.search.lexical.customed_bm25 import CustomBM25

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#### Download SciFact dataset and unzip it
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Load SciFact data using BEIR's GenericDataLoader
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Elasticsearch connection settings
hostname = "http://localhost:9200"
index_name = "custom_bm25_scifact"  # Index created via curl with custom BM25

#### Initialize CustomBM25 with Elasticsearch
bm25_model = CustomBM25(index_name=index_name, hostname=hostname, initialize=False)

#### Index the SciFact corpus into Elasticsearch
logging.info("Indexing the SciFact corpus...")
bm25_model.index(corpus=corpus)
logging.info("Indexing completed.")

#### Retrieve documents using Custom BM25
top_k = 10
logging.info(f"Retrieving top-{top_k} documents using Custom BM25 for each query...")
results = bm25_model.search(corpus=corpus, queries=queries, top_k=top_k)

#### Evaluate retrieval performance using BEIR's qrels
from beir.retrieval.evaluation import EvaluateRetrieval
retriever = EvaluateRetrieval(bm25_model)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Show example retrieval results
import random
query_id, scores_dict = random.choice(list(results.items()))
