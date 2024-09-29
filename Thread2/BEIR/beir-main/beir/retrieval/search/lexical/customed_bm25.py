from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import tqdm
import time

class CustomBM25:
    def __init__(self, index_name: str, hostname: str = "localhost", keys: Dict[str, str] = {"title": "title", "body": "txt"}, 
                 language: str = "english", batch_size: int = 128, timeout: int = 100, retry_on_timeout: bool = True, 
                 maxsize: int = 24, number_of_shards: int = "default", initialize: bool = True, sleep_for: int = 2):
        
        self.results = {}
        self.batch_size = batch_size
        self.initialize = initialize
        self.sleep_for = sleep_for
        
        self.config = {
            "hostname": hostname, 
            "index_name": index_name,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize,
            "number_of_shards": number_of_shards,
            "language": language
        }
        
        # Set up Elasticsearch connection
        self.es = Elasticsearch([hostname], timeout=timeout, retry_on_timeout=retry_on_timeout, maxsize=maxsize)
        
        if self.initialize:
            self.initialise()
    
    def initialise(self):
        """Initialize the Elasticsearch index by deleting and recreating it."""
        self.es.indices.delete(index=self.config['index_name'], ignore=[400, 404])  # Ignore errors if index doesn't exist
        time.sleep(self.sleep_for)  # Sleep to avoid issues when deleting/creating index
        self.create_index()
    
    def create_index(self):
        """Create Elasticsearch index with the predefined custom BM25 similarity."""
        # As the index is created via curl with custom BM25, we assume it's already set up
        print(f"Index {self.config['index_name']} is assumed to be created via curl.")
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """Index the corpus into Elasticsearch."""
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        
        # Prepare the bulk actions for indexing
        actions = [
            {
                "_index": self.config['index_name'],
                "_id": doc_id,
                "_source": {
                    self.config['keys']['title']: doc.get("title", ""),
                    self.config['keys']['body']: doc.get("text", "")
                }
            }
            for doc_id, doc in corpus.items()
        ]
        
        # Use Elasticsearch's bulk API to index the documents
        for ok, action in streaming_bulk(self.es, actions=actions):
            progress.update(1)
        progress.reset()
        progress.close()

    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Search the indexed corpus using the custom BM25 algorithm."""
        
        # Index the corpus if not already done
        if self.initialize:
            self.index(corpus)
            time.sleep(self.sleep_for)
        
        # Prepare the queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        for start_idx in tqdm.trange(0, len(query_texts), self.batch_size, desc='Processing queries'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            query_texts_batch = query_texts[start_idx:start_idx+self.batch_size]
            
            # Perform multi-search using Elasticsearch
            search_body = []
            for query_text in query_texts_batch:
                search_body.append({"index": self.config['index_name']})
                search_body.append({
                    "query": {
                        "multi_match": {
                            "query": query_text,
                            "fields": [self.config['keys']['title'], self.config['keys']['body']],
                            "type": "best_fields"
                        }
                    },
                    "size": top_k + 1  # Retrieve top_k results, add 1 extra to avoid including the query itself if exists
                })
        
            results = self.es.msearch(body=search_body)['responses']
            
            # Process the results
            for query_id, result in zip(query_ids_batch, results):
                self.results[query_id] = {
                    hit["_id"]: hit["_score"]
                    for hit in result['hits']['hits'][:top_k]
                }

        return self.results
