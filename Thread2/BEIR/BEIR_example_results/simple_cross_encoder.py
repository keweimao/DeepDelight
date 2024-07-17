from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import pathlib, os
import logging
import random
from elasticsearch import Elasticsearch
from beir.retrieval.search.lexical.elastic_search import ElasticSearch  # Ensure this import is correct

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Provide parameters for elastic-search
hostname = "https://localhost:9200" # localhost
index_name = "reranking" # nfcorpus
initialize = True

# Retrieve the ELASTIC_PASSWORD environment variable
elastic_password = "Mf5wAwVTOwk2VMYnrJwz"

if not elastic_password:
    raise ValueError("The ELASTIC_PASSWORD environment variable is not set.")

# print("Elasticsearch Password:", elastic_password)

# Initialize Elasticsearch with SSL verification
es_client = Elasticsearch(
    [hostname],
    http_auth=('elastic', elastic_password),
    use_ssl=True,
    verify_certs=True,
    ca_certs='C:/Users/24075/http_ca.crt'  # Path to your CA certificate
)

# Update the ElasticSearch class to use the configured es_client
config = {
    "hostname": hostname,
    "index_name": index_name,
    "keys": {"title": "title", "body": "txt"},
    "timeout": 100,
    "retry_on_timeout": True,
    "maxsize": 24,
    "number_of_shards": "default",
    "language": "english"
}

elastic_search = ElasticSearch(config)
elastic_search.es = es_client

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
model.es = elastic_search

retriever = EvaluateRetrieval(model)

#### Retrieve initial results using BM25
bm25_results = retriever.retrieve(corpus, queries)

#### Rerank using Cross-Encoder model
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
reranker = Rerank(cross_encoder_model, batch_size=128)
rerank_results = reranker.rerank(corpus, queries, bm25_results, top_k=100)

#### Evaluate your reranked retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
