from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import ScriptedBM25Search

import pathlib, os, random, json
import logging
from elasticsearch import Elasticsearch

##################### Parameters ########################
# Connection parameters (CHECK EVERYTIME!)
hostname = "https://localhost:9201"
index_name = "deepdelight"
elastic_password = "vZY1LMN41B=xQ3yvB3fF"
elastic_username = "elastic"

# Elastic parameters - LEXICAL BM25
k1 = 1.2
b = 0.75
avgdl = 201.81 # Average length for scifact document
#########################################################

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Lexical Retrieval using scripted BM25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

#### Initialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

# Initialize Elasticsearch with SSL verification
es_client = Elasticsearch(
    [hostname],
    http_auth=(elastic_username, elastic_password),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
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

# Initialize the ScriptedBM25Search class
model = ScriptedBM25Search(index_name=index_name, hostname=hostname, initialize=initialize, k1=k1, b=b, avgdl=avgdl, username=elastic_username, password=elastic_password)

retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
