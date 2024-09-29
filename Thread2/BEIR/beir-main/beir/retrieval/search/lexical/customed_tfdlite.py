from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import tqdm
import time
import logging
from typing import Dict, List

class TFDLITE:
    def __init__(self, 
                 index_name: str, 
                 idl_cube_root: bool, # New parameter to control the iDL 1/3 mode
                 hostname: str = "localhost", 
                 keys: Dict[str, str] = {"title": "title", "body": "txt"}, 
                 language: str = "english", 
                 batch_size: int = 128, 
                 timeout: int = 100, 
                 retry_on_timeout: bool = True, 
                 maxsize: int = 24, 
                 number_of_shards: int = "default", 
                 initialize: bool = True, sleep_for: int = 2):  
        
        self.results = {}
        self.batch_size = batch_size
        self.initialize = initialize
        self.sleep_for = sleep_for
        self.idl_cube_root = idl_cube_root  # Store the mode
        
        # Modify the config keys to match the correct field names in your Elasticsearch index
        self.config = {
            "hostname": hostname, 
            "index_name": index_name,
            "keys": {"title": "title", "body": "text"},  # Ensure the correct fields are used ('txt' instead of 'body')
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
        logging.info(f"Resetting index: {self.config['index_name']}")
        self.es.indices.delete(index=self.config['index_name'], ignore=[400, 404])  # Ignore errors if index doesn't exist
        time.sleep(self.sleep_for)  # Sleep to avoid issues when deleting/creating index
        self.create_index()
    
    def create_index(self):
        """Create the Elasticsearch index with the necessary DLITE similarity settings."""
        
        # Define the script to use for term weighting
        if self.idl_cube_root:
            script_source = """double nt = term.docFreq; double tf = nt; double k1 = 1.2; double b = 0.75; double N = 5183; double ld = doc.length; double avl = 201.81; double norm_tf = tf / (tf + (k1 * (1 - b + b * (ld / avl)))); double qt = nt / (N + 1); double q_prime_t = 1 - qt; double dlite = (q_prime_t / 2) + (1 - qt * (1 - Math.log(qt))) - ((1 - qt * qt * (1 - 2 * Math.log(qt))) / (2 + 2 * qt)); double idl_cube = norm_tf * Math.cbrt(dlite); return idl_cube;"""
        else:
            script_source = """double nt = term.docFreq; double tf = nt; double k1 = 1.2; double b = 0.75; double N = 5183; double ld = doc.length; double avl = 201.81; double norm_tf = tf / (tf + (k1 * (1 - b + b * (ld / avl)))); double qt = nt / (N + 1); double q_prime_t = 1 - qt; double dlite = (q_prime_t / 2) + (1 - qt * (1 - Math.log(qt))) - ((1 - qt * qt * (1 - 2 * Math.log(qt))) / (2 + 2 * qt)); double idl = norm_tf * dlite; return idl;"""
        
        # Create the index with the scripted DLITE similarity
        settings = {
            "settings": {
                "number_of_shards": 1,
                "similarity": {
                    "scripted_dlite": {
                        "type": "scripted",
                        "script": {
                            "source": script_source
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "txt": {
                        "type": "text",
                        "similarity": "scripted_dlite"
                    },
                    "title": {
                        "type": "text",
                        "similarity": "scripted_dlite"
                    }
                }
            }
        }
        self.es.indices.create(index=self.config['index_name'], body=settings)
        logging.info(f"Created index: {self.config['index_name']}")
    
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
        """Search the indexed corpus using the custom DLITE-based TF-IDF algorithm."""
        
        # Reset index before indexing and searching
        self.initialise()

        # Index the corpus
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
                            "fields": [self.config['keys']['title'], self.config['keys']['body']],  # Ensure 'txt' is used
                            "type": "best_fields"
                        }
                    },
                    "size": top_k + 1  # Retrieve top_k results, add 1 extra to avoid including the query itself if exists
                })
    
            # Execute the multi-search
            results = self.es.msearch(body=search_body)['responses']
            
            # Process the results or log an error if hits are missing
            for query_id, result in zip(query_ids_batch, results):
                if 'hits' in result and 'hits' in result['hits']:
                    self.results[query_id] = {
                        hit["_id"]: hit["_score"]
                        for hit in result['hits']['hits'][:top_k]
                    }
                else:
                    # Log the full Elasticsearch response if 'hits' is not present
                    logging.error(f"Query {query_id} failed. Elasticsearch Response: {result}")
                    self.results[query_id] = {}  # Return an empty result for the failed query
    
        return self.results
