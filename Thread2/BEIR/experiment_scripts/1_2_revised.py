import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
import random
from beir.retrieval.search.lexical.customed_bm25 import CustomBM25  # Import custom BM25

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#### Download scifact dataset and unzip the dataset
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
retriever = EvaluateRetrieval(bm25_model)
results = retriever.retrieve(corpus, queries)

#### Rerank top-100 docs using Dense Retriever model (SBERT)
model_name = "msmarco-distilbert-base-v3"  # Pretrained SBERT model
logging.info(f"\nThe SBERT model used is: {model_name}.\n")

model = DRES(models.SentenceBERT(model_name), batch_size=128)
dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1, 3, 5, 10, 100])

#### Rerank the results based on dense retrieval (SBERT) using top-100 BM25 results
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved
top_k = 10
query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank+1}: {doc_id} [{corpus[doc_id].get('title', 'N/A')}] - {corpus[doc_id].get('text', 'N/A')}\n")
