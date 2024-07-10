# 2024 Summer

## Regular Meetings

* [ ] Finalize existing work
* [ ] New research directions
* [ ] Meetings: weekly presentations
* [ ] Due: IEEE Big Data, 9/8/2024; others

## Thread 2 (Lixiao and Mengyang)

### 2024 Summer Objectives

* [ ] 1. Wrap up RAG implementation on NewsQA and QAConv
* [ ] 2. New focus on the retrieval stage with dense vectors
  * [ ] Consider Robust 04 (HARD track): https://paperswithcode.com/dataset/robust04
  * [ ] Other IR benchmark datasets (e.g. HotpotQA, TREC-Covid) at: 
    * [ ] https://paperswithcode.com/dataset/beir, 
    * [ ] https://paperswithcode.com/paper/beir-a-heterogenous-benchmark-for-zero-shot
  * [ ] BEIR code: https://github.com/beir-cellar/beir
    * [x] [Local setup](./Thread2/BEIR/BEIR_local_setup.md)
  * [ ] Results: 
    * [ ] https://github.com/beir-cellar/beir
    * [ ] https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475
  * [ ] Retrieval (scoring) functions to consider
      * [ ] Benchmark: dot, cosine, TF*IDF, BM25
      * [ ] New methods: DLITE, BM25-revision, Portfolio (diversity)

### 7/3/2024

* [ ] Sonia presentation
* [ ] Update on existing RAG implementations and readiness
* [ ] Next steps, setup for research on dense vector retrieval with Elastic
  * [ ] Setup: Set up Elastic locally first, and then on BVM96
    * [ ] Elastic local deployment
    * [ ] Elastic BVM96 deployment
    * [ ] [BEIR Elasticsearch examples](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)
  * [ ] Code: research and reuse BEIR code, with existing Elastic TF*IDF, BM25 baselines
    * [x] [BEIR local setup](./Thread2/BEIR/BEIR_local_setup.md)
    * [x] Test the BEIR environment
  * [ ] Data: benmark datasets to consider, ultimately Robust 04 (HARD track)
  * [ ] Test: test on a small subset to hopefully replicate some baselines

### 7/10/2024

* [ ] Lixiao presentation, potential implication on diversity retrieval
* [ ] Ke's DLITE as alterantive to IDF and BM25


### 7/17/2024

* [ ] Mengyang presentation

### References

* Ke's 2020 Big Data using DLITE as a scoring function: https://ieeexplore.ieee.org/document/10020937

## Thread 4 (Sonia and Michael)

* [ ] NER: to work on loss functions, new data (NY Times)
    * [ ] Done with new BERT (Microsoft) implementation
* [ ] Information representation with DLITE, on RCV1 data
