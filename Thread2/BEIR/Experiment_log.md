# Experiment Log for Customized Retrieval

This note provides details for updates of BEIR packages and remote machine (BVM96) running scripts.\ 
For results running on local machines, refer to:\
[BEIR_local_results.md](./BEIR_local_results.md)

Lixiao Yang \
7/30/2024 \
9/29/2024 [Updated]

## How to run the scripts in the designated docker
Follow the steps for running scripts using customed methods based on the recent updates for the code (9/22/2024):
1. Log in into the BVM96 machine using your credentials
    ```cmd
    ssh your_credential@bvm96.cci.drexel.edu
    ```
2. Examine existing docker containers
    ```cmd
    docker ps -a
    ```
3. Start the `deep_delight` docker which has been setup by @LixiaoYang, you do not need to create a new container anymore
   ```cmd
   docker start deep_delight
   ```
   The container was created using the following command (this is for reference only, you don't need to go through this one):
   ```cmd
   docker run --name deep_delight --net elastic -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -it -m 60GB docker.elastic.co/elasticsearch/elasticsearch:8.14.3
   ```
   `deep_delight` container has disabled security settings since we are running locally, the memory is allocated as 60GB, using elasticsearch version 8.14.3, port 9200.
4. Check the connection in your current window:
   ```cmd
   curl -X GET "http://localhost:9200"
   ```
   If following results are returned, it means the elasticsearch server is started.
   ```cmd
    {
    "name" : "4fa5b7f5f85c",
    "cluster_name" : "docker-cluster",
    "cluster_uuid" : "F-9T4EK5QjOKqb2k3kHcqA",
    "version" : {
        "number" : "8.14.3",
        "build_flavor" : "default",
        "build_type" : "docker",
        "build_hash" : "d55f984299e0e88dee72ebd8255f7ff130859ad0",
        "build_date" : "2024-07-07T22:04:49.882652950Z",
        "build_snapshot" : false,
        "lucene_version" : "9.10.0",
        "minimum_wire_compatibility_version" : "7.17.0",
        "minimum_index_compatibility_version" : "7.0.0"
    },
    "tagline" : "You Know, for Search"
    }
    ```
    Examine the index definition within the elasticsearch:
    ```cmd
    curl -X PUT "http://localhost:9200/deepdelight"
    ```
    Expected return:
    ```cmd
    {"acknowledged":true,"shards_acknowledged":true,"index":"deepdelight"}
    ```

5. Define the customed script using `curl` to create the new logic when start the new index within the elasticsearch service, below is an example using customed BM25 search logic:
   ```cmd
    curl -X PUT "http://localhost:9200/custom_bm25_scifact" -H 'Content-Type: application/json' -d'
    {
    "settings": {
        "number_of_shards": 1,
        "similarity": {
        "scripted_bm25": {
            "type": "scripted",
            "weight_script": {
            "source": "double k1 = 1.2; double b = 0.75; double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0; return query.boost * idf;"
            },
            "script": {
            "source": "double k1 = 1.2; double b = 0.75; double tf = doc.freq; double docLength = doc.length; double avgdl = 201.81; double norm = (1 - b) + b * (docLength / avgdl); return weight * ((tf * (k1 + 1)) / (tf + k1 * norm));"
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
    }'
    ```
    The code creates a new index within `deep_delight` called `custom_bm25_scifact`, remember to use `weight_script` method for and pass the `source` part in painless format (all codes should be defined within one line, no new lines or comments).
6. Check if the customed logic has been passed to elastic and the index is created:
   ```cmd
   curl -X GET "http://localhost:9200/custom_bm25_scifact/_mapping?pretty"
    {
    "custom_bm25_scifact" : {
        "mappings" : {
        "properties" : {
            "content" : {
            "type" : "text",
            "similarity" : "scripted_bm25"
            },
            "title" : {
            "type" : "text",
            "fields" : {
                "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
                }
              }
            },
            "txt" : {
            "type" : "text",
            "fields" : {
                "keyword" : {
                "type" : "keyword",
                "ignore_above" : 256
                }
              }
            }
          }
        }
      }
    }
    ```
    Check for the script:
    ```cmd
    curl -X GET "http://localhost:9200/custom_bm25_scifact/_settings?pretty"
    {
      "custom_bm25_scifact" : {
        "settings" : {
          "index" : {
            "routing" : {
              "allocation" : {
                "include" : {
                  "_tier_preference" : "data_content"
                }
              }
            },
            "number_of_shards" : "1",
            "provided_name" : "custom_bm25_scifact",
            "similarity" : {
              "scripted_bm25" : {
                "type" : "scripted",
                "weight_script" : {
                  "source" : "double k1 = 1.2; double b = 0.75; double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0; return query.boost * idf;"
                },
                "script" : {
                  "source" : "double k1 = 1.2; double b = 0.75; double tf = doc.freq; double docLength = doc.length; double avgdl = 201.81; double norm = (1 - b) + b * (docLength / avgdl); return weight * ((tf * (k1 + 1)) / (tf + k1 * norm));"
                }
              }
            },
            "creation_date" : "1727044456099",
            "number_of_replicas" : "1",
            "uuid" : "jJmTz3BfQS2lBOvOwIZXXw",
            "version" : {
              "created" : "8505000"
            }
          }
        }
      }
    }
    ```
    Check for field mapping:
    ```cmd
    curl -X GET "http://localhost:9200/custom_bm25_scifact/_mapping/field/title,body?pretty"
    {
      "custom_bm25_scifact" : {
        "mappings" : {
          "title" : {
            "full_name" : "title",
            "mapping" : {
              "title" : {
                "type" : "text",
                "fields" : {
                  "keyword" : {
                    "type" : "keyword",
                    "ignore_above" : 256
                  }
                }
              }
            }
          }
        }
      }
    }
    ```
7. Run the related script in the related path, remember to switch to `beir` environment and use proper logging for your running results:
   ```cmd
   (beir) ly364@bvm96:~/DeepDelight/BEIR$ python 1_1_revised.py | tee result_logs/1_1_revised_log.txt
   ```

## Update for customized BM25 method

### BEIR code updates

#### Add scripted BM25 method within beir [Outdated]
9/22/2024 Comment: This method is stale, refer to the next seciton.\
<del>Update for customized BM25 method in `beir-main/beir/retrieval/search/lexical/__init__.py`:<del>
```python
from .scripted_bm25 import ScriptedBM25Search
```

#### Add an elastic search script [Outdated]
9/29/2024 Comment: This method is stale, by defining the logic when creating the elastic index, we can use the original elastic search methods in BEIR.\
<del>[custom_elastic_search.py](./beir-main/beir/retrieval/search/lexical/customed_elastic_search.py)\
Modified based on [`beir-main/beir/retrieval/search/lexical/elastic_search.py`](./beir-main/beir/retrieval/search/lexical/elastic_search.py):<del>
- <del>For `create_index` method, add an additional parameter `body` to handle the scripted BM25 method (custom BM25 method)
- <del>Error and exception handling
To use the method:
```python
from beir.retrieval.search.lexical.customed_elastic_search import CustomedElasticSearch
```


#### Add an custom bm25 search script [Updated at 9/22/2024]
[customed_bm25.py](./beir-main/beir/retrieval/search/lexical/customed_bm25.py) \
Modified based on [`beir-main/beir/retrieval/search/lexical/bm25_search.py`](./beir-main/beir/retrieval/search/lexical/bm25_search.py):
- <del>Works together with `customed_elastic_search.py` to add custom BM25 retrieval logic<del>
- Error and exception handling
To use the method:
```python
from beir.retrieval.search.lexical.customed_bm25 import CustomBM25
```

### 07/30/2024 Experiment scripts update [Outdated]

**Customed BM25 lexical retrieval**\
[1_1_custom_bm25_lexical.py](./experiment_scripts/1_1_custom_bm25_lexical.py) \
Modified BM25 lexical retrieval code using `CustomedElasticSearch` and `ScriptedBM25Search`. \
Parameters:
```python
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
```

7/30/2024: **Lexical BM25 results running on BVM96 indicating same results as the elastic embedded BM25 method.**

**Customed BM25 reranking with Sentense-BERT**\
[1_3_custom_bm25_reranking_sbert.py](./experiment_scripts/1_3_custom_bm25_reranking_sbert.py) \
Modified BM25 reranking with Sentence-BERT code using `CustomedElasticSearch` and `ScriptedBM25Search`. \
Parameters:
```python
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
```

7/30/2024: **Reranking with SBERT BM25 results running on BVM96 indicating same results as the elastic embedded BM25 method.**

**Customed BM25 reranking with Cross-Encoder**\
[1_2_custom_bm25_reranking_ce.py](./experiment_scripts/1_2_custom_bm25_reranking_ce.py) \ 
Modified BM25 reranking with Cross-Encoder code using `CustomedElasticSearch` and `ScriptedBM25Search`. \
Parameters:
```python
##################### Parameters ########################
# Connection parameters (CHECK EVERYTIME!)
hostname = "https://localhost:9201"
index_name = "deepdelight"
elastic_password = "vZY1LMN41B=xQ3yvB3fF"
elastic_username = "elastic"

# Elastic parameters - RERANKING W/ SENTENCE-BERT
k1 = 1.2
b = 0.75
avgdl = 201.81 # Average length for scifact document
#########################################################
```

7/30/2024: **Reranking with Cross-Encoder BM25 results running on BVM96 indicating same results as the elastic embedded BM25 method.**

### 9/22/2024 Experiment scripts update
#### 1. Updated customed BM25 running scripts
[1_1_revised.py](./experiment_scripts/1_1_revised.py) is the code using customed BM25 method for lexical retrieval.\
[1_2_revised.py](./experiment_scripts/1_2_revised.py) is the code using customed BM25 method for SBERT reranking retrieval.\
[1_3_revised.py](./experiment_scripts/1_3_revised.py) is the code using customed BM25 method for cross-encoder reranking retrieval.

Custom BM25 logic definition:
```json
curl -X PUT "http://localhost:9200/custom_bm25_scifact" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "number_of_shards": 1,
    "similarity": {
      "scripted_bm25": {
        "type": "scripted",
        "weight_script": {
          "source": "double k1 = 1.2; double b = 0.75; double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0; return query.boost * idf;"
        },
        "script": {
          "source": "double k1 = 1.2; double b = 0.75; double tf = doc.freq; double docLength = doc.length; double avgdl = 10; double norm = (1 - b) + b * (docLength / avgdl); return weight * ((tf * (k1 + 1)) / (tf + k1 * norm));"
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
}'
```

:warning:Note:\
1. It is recommened to set `initializer=Ture` when running the code which would enable the reset at everyrun. So that previous results/embeddings would not affect the following one.
2. The index name and port name should align with the defined logic index name and current docker container port number.
3. The password part is deleted due to disabled security setting.

#### 2. Customed BM25 running results
To aviod the same mistakes as the previous custom BM25 methods, different parameters are changed in different runs to see if the BM25 retrieval logic has been passed to elastic and the results have difference. The results for customed BM25 methods are shown as follows:
| Metric     | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@100 | MAP@1  | MAP@3  | MAP@5  | MAP@10 | MAP@100 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@100 | P@1   | P@3   | P@5   | P@10  | P@100 |
|------------|--------|--------|--------|---------|----------|--------|--------|--------|--------|---------|----------|----------|----------|-----------|------------|-------|-------|-------|-------|-------|
| Benchmark  | 0.5267 | 0.6077 | 0.6308 | 0.6537  | 0.6537   | 0.5082 | 0.5813 | 0.5955 | 0.6064 | 0.6064  | 0.5082   | 0.6623   | 0.7184   | 0.7843    | 0.7843     | 0.5267 | 0.2344 | 0.1547 | 0.0863 | 0.0086 |
| Short avgdl (avgdl=10) | 0.1533 | 0.1644 | 0.169  | 0.1718  | 0.1718   | 0.1422 | 0.1572 | 0.1597 | 0.1608 | 0.1608  | 0.1422   | 0.1756   | 0.1862   | 0.1946    | 0.1946     | 0.1533 | 0.0622 | 0.04   | 0.021  | 0.0021 |
| Larger b (b=0.9) | 0.24   | 0.2726 | 0.2793 | 0.286   | 0.286    | 0.2261 | 0.2601 | 0.264  | 0.2665 | 0.2665  | 0.2261   | 0.2964   | 0.3131   | 0.3327    | 0.3327     | 0.24   | 0.1078 | 0.068  | 0.0367 | 0.0037 |
| Smaller b (b=0.5) | 0.2033 | 0.2193 | 0.225  | 0.2295  | 0.2295   | 0.1944 | 0.211  | 0.2144 | 0.2166 | 0.2166  | 0.1944   | 0.2331   | 0.2457   | 0.2584    | 0.2584     | 0.2033 | 0.0822 | 0.0533 | 0.0283 | 0.0028 |
| Larger k1 (k1=1.5) | 0.4667 | 0.5234 | 0.5437 | 0.5571  | 0.5571   | 0.4462 | 0.5014 | 0.5136 | 0.5195 | 0.5195  | 0.4462   | 0.5648   | 0.6165   | 0.6549    | 0.6549     | 0.4667 | 0.2033 | 0.1333 | 0.072  | 0.0072 |
| Smaller k1 (k1=1.05) | 0.46   | 0.5266 | 0.5443 | 0.5633  | 0.5633   | 0.4417 | 0.505  | 0.5151 | 0.5243 | 0.5243  | 0.4417   | 0.5733   | 0.6144   | 0.6683    | 0.6683     | 0.46   | 0.2044 | 0.1333 | 0.074  | 0.0074 |


[Raw results log files location](./result_logs/) with sequence numbers 1_1 to 1_3.

**Conclution:**
1. Different parameters result in different retrieval evaluation metrics
2. Parameters' behavior aligns with expected directions
3. We can conlude that the **newer ways to define the customed BM25 logic is effective**.

### 9/29/2024 Experiment scripts update
#### 1. Updated customed TF-DLITE(iDL) running scirpts
[customed_tfdlite.py](./beir-main/beir/retrieval/search/lexical/customed_tfdlite.py) is the customed $iDL$ methods that replace the inverse term frequency in traditional TDIDF method. The `TFDLITE` class are defined as:
```python
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
                 initialize: bool = True, 
                 sleep_for: int = 2):  
```
The method can be imported as:
```python
from beir.retrieval.search.lexical.customed_tfdlite import TFDLITE
```


[2_1_idl_lexical.py](./experiment_scripts/2_1_idl_lexical.py) is the code using customed TF-DLITE for lexical retrieval.\
[2_2_idl_reranking_sbert.py](./experiment_scripts/2_2_idl_reranking_sbert.py) is the code using customed TF-DLITE for lexical retrieval.\
[2_3_idl_reranking_ce.py](./experiment_scripts/2_3_idl_reranking_ce.py) is the code using customed TF-DLITE for lexical retrieval.

Customed $TF-DLITE$ ($iDL$) definition:
```json
{
  "settings": {
    "number_of_shards": 1,
    "similarity": {
      "scripted_dlite": {
        "type": "scripted",
        "script": {
          "source": "double nt = term.docFreq; double tf = nt; double k1 = 1.2; double b = 0.75; double N = 5183; double ld = doc.length; double avl = 201.81; double norm_tf = tf / (tf + (k1 * (1 - b + b * (ld / avl)))); double qt = nt / N; double q_prime_t = 1 - qt; double dlite = (q_prime_t / 2) + (1 - qt * (1 - Math.log(qt))) - ((1 - qt * qt * (1 - 2 * Math.log(qt))) / (2 + 2 * qt)); double idl = norm_tf * dlite; return idl;"
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
}'
```
Customed $TF-DLITE^{1/3}$ ($iDL^{1/3}$) definition:
```json
{
  "settings": {
    "number_of_shards": 1,
    "similarity": {
      "scripted_dlite": {
        "type": "scripted",
        "script": {
          "source": "double nt = term.docFreq; double tf = nt; double k1 = 1.2; double b = 0.75; double N = 5183; double ld = doc.length; double avl = 201.81; double norm_tf = tf / (tf + (k1 * (1 - b + b * (ld / avl)))); double qt = nt / (N + 1); double q_prime_t = 1 - qt; double dlite = (q_prime_t / 2) + (1 - qt * (1 - Math.log(qt))) - ((1 - qt * qt * (1 - 2 * Math.log(qt))) / (2 + 2 * qt)); double idl_cube = norm_tf * Math.cbrt(dlite); return idl_cube;"
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
}'
```

:warning:Note:
1. Pay attention to `idl_cube_root` selection, different boolean values are mapped with different method.
2. It is recommened to set `initializer=Ture` when running the code which would enable the reset at everyrun. So that previous results/embeddings would not affect the following one.
3. The index name and port name should align with the defined logic index name and current docker container port number.
4. The password part is deleted due to disabled security setting.

#### 2. Running results
| Method                    | Retrieval               | Similarity          | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@100 | MAP@1  | MAP@3  | MAP@5  | MAP@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | P@1   | P@3   | P@5   | P@10  |
|---------------------------|-------------------------|---------------------|--------|--------|--------|---------|----------|--------|--------|--------|--------|----------|----------|----------|-----------|-------|-------|-------|-------|
| $iDL$                       | Lexical                 | Cosine Similarity    | 0.5300 | 0.6144 | 0.6320 | 0.6521  | 0.6755   | 0.5115 | 0.5871 | 0.5982 | 0.6081 | 0.5115   | 0.6707   | 0.7134   | 0.7709    | 0.5300 | 0.2378 | 0.1540 | 0.0850 |
| $iDL$                       | Lexical                 | Dot Product    | 0.5300 | 0.6144 | 0.6320 | 0.6521  | 0.6755   | 0.5115 | 0.5871 | 0.5982 | 0.6081 | 0.5115   | 0.6707   | 0.7134   | 0.7709    | 0.5300 | 0.2378 | 0.1540 | 0.0850 |
| $iDL^{1/3}$           | Lexical                 | Cosine Similarity    | 0.5300 | 0.6144 | 0.6320 | 0.6521  | 0.6755   | 0.5115 | 0.5871 | 0.5982 | 0.6081 | 0.5115   | 0.6707   | 0.7134   | 0.7709    | 0.5300 | 0.2378 | 0.1540 | 0.0850 |
| $iDL$                       | SBERT Reranking         | Cosine Similarity    | 0.4233 | 0.4842 | 0.5104 | 0.5369  | 0.5756   | 0.3994 | 0.4593 | 0.4768 | 0.4886 | 0.3994   | 0.5256   | 0.5887   | 0.6690    | 0.4233 | 0.1933 | 0.1333 | 0.0753 |
| $iDL$                       | SBERT Reranking         | Dot Product          | 0.4000 | 0.4583 | 0.4778 | 0.5081  | 0.5492   | 0.3773 | 0.4329 | 0.4465 | 0.4602 | 0.3773   | 0.5018   | 0.5504   | 0.6401    | 0.4000 | 0.1844 | 0.1253 | 0.0727 |
| $iDL^{1/3}$             | SBERT Reranking         | Cosine Similarity    | 0.4233 | 0.4842 | 0.5104 | 0.5369  | 0.5756   | 0.3994 | 0.4593 | 0.4768 | 0.4886 | 0.3994   | 0.5256   | 0.5887   | 0.6690    | 0.4233 | 0.1933 | 0.1333 | 0.0753 |
| $iDL$                       | Cross-Encoder           | Cosine Similarity    | 0.5700 | 0.6340 | 0.6508 | 0.6717  | 0.6935   | 0.5453 | 0.6073 | 0.6198 | 0.6296 | 0.5453   | 0.6816   | 0.7216   | 0.7838    | 0.5700 | 0.2456 | 0.1600 | 0.0880 |
| $iDL$                       | Cross-Encoder           | Dot Product          | 0.5700 | 0.6340 | 0.6508 | 0.6717  | 0.6935   | 0.5453 | 0.6073 | 0.6198 | 0.6296 | 0.5453   | 0.6816   | 0.7216   | 0.7838    | 0.5700 | 0.2456 | 0.1600 | 0.0880 |
| $iDL^{1/3}$             | Cross-Encoder           | Cosine Similarity    | 0.5700 | 0.6340 | 0.6508 | 0.6717  | 0.6935   | 0.5453 | 0.6073 | 0.6198 | 0.6296 | 0.5453   | 0.6816   | 0.7216   | 0.7838    | 0.5700 | 0.2456 | 0.1600 | 0.0880 |

[Raw results log files location](./result_logs/) with sequence numbers 2_1 to 2_3.