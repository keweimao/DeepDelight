"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random
import logging

# Added for Lixiao's local machine
from beir.retrieval.search.lexical.elastic_search import ElasticSearch
from elasticsearch import Elasticsearch

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

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "https://localhost:9201" #localhost
index_name = "tfidf" # nfcorpus

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

######### Added for Lixiao's local machine #########
elastic_password = "ApN3T=skh6Pl3jki82V1"

# Initialize Elasticsearch with SSL verification
es_client = Elasticsearch(
    [hostname],
    http_auth=('elastic', elastic_password),
    use_ssl=False,
    verify_certs=False,
    ca_certs='C:/Users/24075/http_ca.crt'  # Path to your CA certificate
)

# Custom BM25 Similarity Settings
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "similarity": {
            "scripted_bm25": {
                "type": "scripted",
                "weight_script": {
                    "source": """
                        double k1 = 1.2;
                        double b = 0.75;
                        double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0;
                        return query.boost * idf;
                    """
                },
                "script": {
                    "source": """
                        double k1 = 1.2;
                        double b = 0.75;
                        double tf = doc.freq;
                        double docLength = doc.length;
                        double avgdl = 100.0;
                        double norm = (1 - b) + b * (docLength / avgdl);
                        return weight * ((tf * (k1 + 1)) / (tf + k1 * norm));
                    """,
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "similarity": "scripted_bm25"
            }
        }
    }
}

# Create the index with custom BM25 similarity
if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name)
es_client.indices.create(index=index_name, body=index_settings)

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

###############################

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

###### Added for Lixiao's local machine ######
model.es = elastic_search
###########################################

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)
print(f"这是results的类型{type(results)}")

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