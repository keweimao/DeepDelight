from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical.scripted_bm25 import ScriptedBM25Search
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

import pathlib, os
import logging
import random
from elasticsearch import Elasticsearch

##################### Parameters ########################
# Connection parameters (CHECK EVERYTIME!)
hostname = "https://localhost:9201"
index_name = "deepdelight"
elastic_password = "vZY1LMN41B=xQ3yvB3fF"
elastic_username = "elastic"

# Elastic parameters - RERANKING W/ CROSS-ENCODERS
k1 = 1.2
b = 0.75
avgdl = 201.81 # Average length for scifact document
k_values = [1, 3, 5, 10, 100]
#########################################################

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
hostname = "https://localhost:9201" #localhost
index_name = "deepdelight" # nfcorpus
initialize = True

elastic_password = "vZY1LMN41B=xQ3yvB3fF"
elastic_username = "elastic"

# Initialize Elasticsearch with SSL verification
es_client = Elasticsearch(
    [hostname],
    http_auth=(elastic_username, elastic_password),
    use_ssl=False,
    verify_certs=False,
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
    "number_of_shards": 1,
    "language": "english",
    "username": elastic_username,
    "password": elastic_password
}

model = ScriptedBM25Search(index_name=index_name, hostname=hostname, initialize=initialize, k1=1.2, b=0.75, avgdl=201.81, username=elastic_username, password=elastic_password)

retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Reranking top-100 docs using Dense Retriever model 
model_name = "msmarco-distilbert-base-v3" # msmarco-distilbert-base-v3
print(f"\nThe SBERT model used is: {model_name}.\n")
model = DRES(models.SentenceBERT(model_name), batch_size=128)
dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=k_values) # cos_sim, dot_score

#### Retrieve dense results (format of results is identical to qrels)
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
