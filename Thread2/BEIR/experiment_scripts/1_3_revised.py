import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
from beir.retrieval.search.lexical.customed_bm25 import CustomBM25  # Import custom BM25

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#### Download SciFact dataset and unzip the dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Load SciFact data using BEIR's GenericDataLoader
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#########################################
#### (1) RETRIEVE Top-100 docs using Custom BM25
#########################################

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

################################################
#### (2) RERANK Top-100 docs using Cross-Encoder
################################################

#### Cross-Encoder model options
# You can change this to other models like 'cross-encoder/ms-marco-TinyBERT-L-6', 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
logging.info(f"\nThe cross encoder model used is: {cross_encoder_name}.\n")
cross_encoder_model = CrossEncoder(cross_encoder_name)

#### Initialize Reranker with Cross-Encoder
reranker = Rerank(cross_encoder_model, batch_size=128)

#### Rerank the top-100 documents using Cross-Encoder
rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

#### Evaluate retrieval and reranking performance using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved after reranking
top_k = 10
query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank+1}: {doc_id} [{corpus[doc_id].get('title', 'N/A')}] - {corpus[doc_id].get('text', 'N/A')}\n")
