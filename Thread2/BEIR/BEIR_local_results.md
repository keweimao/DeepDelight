# BEIR Local Running Results
Lixiao Yang\
7/16/2024

[BEIR Available Datasets](https://github.com/beir-cellar/beir/wiki/Datasets-available)\
[BEIR Available Models](https://github.com/beir-cellar/beir/wiki/Models-available)\
[BEIR Examples](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)\
[Leaderboard](https://eval.ai/web/challenges/challenge-page/1897/leaderboard) (Ranked by nDCG@10)

## Summary
### 7/16/2024
- Based on the nDCG results (used by the scifact leaderboard), the BM25 number is close to the record
- Reranking using Sentence Transformer shows relative bad results, with very low nDCF for base-bert model
- Reranking using cross encoder shows better results (close to benchmark results), follows the trend that larger models have better results


## Running Configuration
### Dataset-01
Dataset: Scifact [Homepage](https://github.com/allenai/scifact)\
[Scifact Benchmark Leaderboard](https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475/SciFact)

## Lexical Retrieval
### Dataset-01
**BM25 Evaluation Metrics**  [Code](./BEIR_example_results/lexical_bm25.py)\
[Output Log](./BEIR_example_results/lexical_bm25_result.txt)
| k   | nDCG   | MAP    | Recall | Precision |
| --- | ------ | ------ | ------ | --------- |
| 1   | 0.5400 | 0.5193 | 0.5193 | 0.5400    |
| 3   | 0.6155 | 0.5896 | 0.6690 | 0.2378    |
| 5   | 0.6347 | 0.6018 | 0.7154 | 0.1560    |
| 10  | 0.6563 | 0.6119 | 0.7790 | 0.0853    |
| 100 | 0.6810 | 0.6179 | 0.8842 | 0.0100    |
| 1000| 0.6915 | 0.6183 | 0.9683 | 0.0011    |


## Reranking
[Pretrained SentenceTransformer Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)

- Something to consider: for now BEIR only supports cosine similarity and dot score for scoring functions in evaluation retrieval process.

### Dataset-01
**Reranking top-100 BM25 results with SBERT CE**  [Code](./BEIR_example_results/evaluate_bm25_sbert_reranking.py)\
Model: msmarco-distilbert-base-v3 [Output Log](./BEIR_example_results/evaluate_bm25_sbert_reranking_result.txt)
| k    | NDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.4233 | 0.3994 | 0.3994 | 0.4233    |
| 3    | 0.4858 | 0.4605 | 0.5289 | 0.1944    |
| 5    | 0.5106 | 0.4771 | 0.5887 | 0.1333    |
| 10   | 0.5391 | 0.4895 | 0.6757 | 0.0760    |
| 100  | 0.5768 | 0.4979 | 0.8493 | 0.0096    |
| 1000 | 0.5768 | 0.4979 | 0.8493 | 0.0010    |

Model: bert-base-uncased [Output Log](./BEIR_example_results/evaluate_bm25_sbert_reranking_result2.txt)
| k    | NDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.0733 | 0.0686 | 0.0686 | 0.0733    |
| 3    | 0.0983 | 0.0893 | 0.1161 | 0.0422    |
| 5    | 0.1146 | 0.0988 | 0.1556 | 0.0347    |
| 10   | 0.1337 | 0.1066 | 0.2108 | 0.0240    |
| 100  | 0.1871 | 0.1149 | 0.4884 | 0.0054    |
| 1000 | 0.1871 | 0.1149 | 0.4884 | 0.0005    |



## Reranking with Cross Encoder
[SBERT Cross Encoder Models](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)

### Dataset-01
**Reranking top-100 BM25 results with Dense Retriever**  [Code](./BEIR_example_results/evaluate_bm25_ce_reranking.py)\

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 [Output Log](./BEIR_example_results/evaluate_bm25_ce_reranking_result.txt)
| k    | nDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.5767 | 0.5519 | 0.5519 | 0.5767    |
| 3    | 0.6406 | 0.6140 | 0.6883 | 0.2478    |
| 5    | 0.6600 | 0.6278 | 0.7349 | 0.1627    |
| 10   | 0.6824 | 0.6383 | 0.8011 | 0.0900    |
| 100  | 0.7025 | 0.6437 | 0.8842 | 0.0100    |
| 1000 | 0.7025 | 0.6437 | 0.8842 | 0.0010    |

Model: cross-encoder/ms-marco-MiniLM-L-4-v2 [Output Log](./BEIR_example_results/evaluate_bm25_ce_reranking_result3.txt)
| k    | NDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.5633 | 0.5384 | 0.5384 | 0.5633    |
| 3    | 0.6348 | 0.6085 | 0.6823 | 0.2467    |
| 5    | 0.6590 | 0.6250 | 0.7418 | 0.1640    |
| 10   | 0.6803 | 0.6349 | 0.8038 | 0.0903    |
| 100  | 0.6996 | 0.6400 | 0.8842 | 0.0100    |
| 1000 | 0.6996 | 0.6400 | 0.8842 | 0.0010    |

Model: cross-encoder/ms-marco-TinyBERT-L-2-v2 [Output Log](./BEIR_example_results/evaluate_bm25_ce_reranking_result4.txt)
| k    | NDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.5633 | 0.5326 | 0.5326 | 0.5633    |
| 3    | 0.6222 | 0.5967 | 0.6636 | 0.2411    |
| 5    | 0.6428 | 0.6121 | 0.7131 | 0.1580    |
| 10   | 0.6625 | 0.6212 | 0.7709 | 0.0870    |
| 100  | 0.6892 | 0.6283 | 0.8842 | 0.0100    |
| 1000 | 0.6892 | 0.6283 | 0.8842 | 0.0010    |

Model: cross-encoder/ms-marco-TinyBERT-L-6 [Output Log](./BEIR_example_results/evaluate_bm25_ce_reranking_result2.txt)
| k    | NDCG   | MAP    | Recall | Precision |
| ---- | ------ | ------ | ------ | --------- |
| 1    | 0.5133 | 0.4909 | 0.4909 | 0.5133    |
| 3    | 0.5984 | 0.5701 | 0.6555 | 0.2356    |
| 5    | 0.6209 | 0.5855 | 0.7072 | 0.1580    |
| 10   | 0.6452 | 0.5966 | 0.7798 | 0.0877    |
| 100  | 0.6702 | 0.6032 | 0.8842 | 0.0100    |
| 1000 | 0.6702 | 0.6032 | 0.8842 | 0.0010    |