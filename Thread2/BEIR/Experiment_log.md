# Experiment Log for Customized Retrieval

This note provides details for updates of BEIR packages and remote machine (BVM96) running scripts. \ 
For results running on local machines, refer to:\
[BEIR_local_results.md](./BEIR_local_results.md)

Lixiao Yang \
7/30/2024

## Update for customized BM25 method

### 1. BEIR code updates

**Add scripted BM25 method within beir** \
Update for customized BM25 method in `beir-main/beir/retrieval/search/lexical/__init__.py`:
```python
from .scripted_bm25 import ScriptedBM25Search
```

**Add an elastic search script**\
[custom_elastic_search.py](./beir-main/beir/retrieval/search/lexical/customed_elastic_search.py)\
Modified based on [`beir-main/beir/retrieval/search/lexical/elastic_search.py`](./beir-main/beir/retrieval/search/lexical/elastic_search.py):
- For `create_index` method, add an additional parameter `body` to handle the scripted BM25 method (custom BM25 method)
- Error and exception handling
To use the method:
```python
from beir.retrieval.search.lexical.customed_elastic_search import CustomedElasticSearch
```

**Add an custom bm25 search script** \
[scripted_bm25.py](./beir-main/beir/retrieval/search/lexical/scripted_bm25.py) \
Modified based on [`beir-main/beir/retrieval/search/lexical/bm25_search.py`](./beir-main/beir/retrieval/search/lexical/bm25_search.py):
- Works together with `customed_elastic_search.py` to add custom BM25 retrieval logic
- Error and exception handling
To use the method:
```python
from beir.retrieval.search.lexical.scripted_bm25 import ScriptedBM25Search
```

### 2. Experiment scripts update

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
