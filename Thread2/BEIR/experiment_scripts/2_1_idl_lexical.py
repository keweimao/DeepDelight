import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical.customed_tfdlite import TFDLITE  # Import renamed custom TFDLITE class

#### Set up logging to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

"""
#### Fix the random seed for reproducibility
import numpy as np
np.random.seed(123)
"""

#### Download the SciFact dataset and unzip it
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Load SciFact data using BEIR's GenericDataLoader
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#########################################
#### (1) RETRIEVE Top-k docs using TFDLITE
#########################################

#### Elasticsearch connection settings
hostname = "http://localhost:9200"
index_name = "custom_idl"  # Index created via curl with DLITE similarity

#### Initialize TFDLITE class with Elasticsearch connection
tfdlite_model = TFDLITE(index_name=index_name, hostname=hostname, initialize=True, idl_cube_root=False)


#### Index the SciFact corpus into Elasticsearch (only if not already indexed)
logging.info("Indexing the SciFact corpus...")
tfdlite_model.index(corpus=corpus)  # Indexes the corpus into the custom DLITE-based index
logging.info("Indexing completed.")


#### Retrieve documents using TFDLITE
retriever = EvaluateRetrieval(tfdlite_model, score_function="dot")
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K, Recall@K, and Precision@K
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

"""
#### Fix a deterministic query by using a sorted list of query IDs
query_ids_sorted = sorted(list(results.keys()))  # Sorted list for deterministic selection
query_id = query_ids_sorted[0]  # Select the first query ID for consistency

ranking_scores = results[query_id]
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(min(top_k, len(scores_sorted))):
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