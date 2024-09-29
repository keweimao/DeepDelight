import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.search.lexical.customed_tfdlite import TFDLITE  # Import renamed custom TFDLITE class

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
index_name = "custom_idl" 

#### Initialize TFDLITE class with Elasticsearch connection
idl_cube_root = True
tfdlite_model = TFDLITE(index_name=index_name, hostname=hostname, initialize=True, idl_cube_root=idl_cube_root)
logging.info(f"iDL Settings: Index name: {index_name}, Hostname: {hostname}, iDL cube root?: {idl_cube_root}\n")

#### Index the SciFact corpus into Elasticsearch
logging.info("Indexing the SciFact corpus...")
tfdlite_model.index(corpus=corpus)
logging.info("Indexing completed.")

#### Retrieve documents using TFDLITE
retriever = EvaluateRetrieval(tfdlite_model, score_function="dot")
results = retriever.retrieve(corpus, queries)

#### Rerank top-100 docs using Dense Retriever model (SBERT)
model_name = "msmarco-distilbert-base-v3"  # Pretrained SBERT model
logging.info(f"\nThe SBERT model used is: {model_name}.\n")

model = DRES(models.SentenceBERT(model_name), batch_size=128)
dense_retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1, 3, 5, 10, 100])

#### Rerank the results based on dense retrieval (SBERT) using top-100 TDFLITE results
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved
top_k = 10

"""
query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank+1}: {doc_id} [{corpus[doc_id].get('title', 'N/A')}] - {corpus[doc_id].get('text', 'N/A')}\n")
"""
#### Specify the query ID to be used (modify this ID for specific queries)
specific_query_id = "1395"  # Replace this with the query ID of interest

if specific_query_id in results:
    ranking_scores = results[specific_query_id]
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Query : {queries[specific_query_id]}\n")

    for rank in range(min(top_k, len(scores_sorted))):
        doc_id = scores_sorted[rank][0]
        score = scores_sorted[rank][1]
        # Format: Rank x: ID [Title] Body [Score]
        logging.info(f"Rank {rank+1}: {doc_id} [{corpus[doc_id].get('title', 'N/A')}] - {corpus[doc_id].get('text', 'N/A')} (Score: {score})\n")
else:
    logging.error(f"Query ID {specific_query_id} not found in the retrieval results.")
