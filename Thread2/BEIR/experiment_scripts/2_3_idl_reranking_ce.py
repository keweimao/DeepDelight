import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import random
from beir.retrieval.search.lexical.customed_tfdlite import TFDLITE

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
#### (1) RETRIEVE Top-100 docs using TFDLITE
#########################################

#### Elasticsearch connection settings
hostname = "http://localhost:9200"
index_name = "custom_idl"

#### Initialize TFDLITE class with Elasticsearch connection
tfdlite_model = TFDLITE(index_name=index_name, hostname=hostname, initialize=True, idl_cube_root=True)

#### Index the SciFact corpus into Elasticsearch
logging.info("Indexing the SciFact corpus...")
tfdlite_model.index(corpus=corpus)
logging.info("Indexing completed.")

#### Retrieve documents using TFDLITE
retriever = EvaluateRetrieval(tfdlite_model, score_function="cos_sim")
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
