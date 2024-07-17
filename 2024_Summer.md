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

* [x] Sonia presentation
* [ ] Update on existing RAG implementations and readiness
    * [x] Adjusted system implemented with lancedb on QAConv dataset:
      * [x] [Scripts](./Thread2/Running_Scripts/qaconv_lancedb.ipynb)
      * [x] [Logs](./Thread2/Running_Scripts/combined_chunks3.log)
    * [x] Local test for `hotpot` dataset based on the former system structure:
      * [x] [Scripts](./Thread2/HOTPOT/QA_hotpot.ipynb)
      * [x] [Results](./Thread2/HOTPOT/output_hotpot.csv)      
* [ ] Next steps, setup for research on dense vector retrieval with Elastic
  * [ ] Setup: Set up Elastic locally first, and then on BVM96
    * [x] Elastic local deployment in Docker
    * [ ] Elastic BVM96 deployment
  * [ ] Code: research and reuse BEIR code, with existing Elastic TF*IDF, BM25 baselines
    * [x] [BEIR local setup](./Thread2/BEIR/BEIR_local_setup.md)
    * [x] Test the BEIR environment
  * [ ] Data: benmark datasets to consider, ultimately Robust 04 (HARD track)
  * [ ] Test: test on a small subset to hopefully replicate some baselines
    * [x] Local test for `scifact` dataset: lexical, reranking and reranking w/ cross encoder
      * [x] [Scripts and Logs](./Thread2/BEIR/BEIR_example_results)
      * [x] [Local Results Summary](./Thread2/BEIR/BEIR_local_results.md) along with evaluation comparison



### 7/10/2024

### 7/17/2024

* [ ] Lixiao presentation, potential implication on diversity retrieval
* [ ] Ke's DLITE as alterantive to IDF and BM25
* [ ] Reading: https://ar5iv.labs.arxiv.org/html/2311.18503
* [ ] Hybrid Dense + Sparse: https://ar5iv.labs.arxiv.org/html/2401.04055


### 7/24/2024

* [ ] Mengyang presentation
* [ ] DLITE reading: 
  * [ ] Information theory of IDF: https://youtu.be/fIIuyorK7BY?si=1uYindxv8jYXISA2
  * [ ] DLITE for IR: https://youtu.be/qQXCgmX8sOk?si=bpcrdeW6HLujIAY4


### References

* Ke's 2020 BigD ata using DLITE as a scoring function: https://ieeexplore.ieee.org/document/10020937

## Thread 4 (Sonia and Michael)

* [ ] NER: to work on loss functions, new data (NY Times)
    * [ ] Done with new BERT (Microsoft) implementation
* [ ] Information representation with DLITE, on RCV1 data
* [ ] Comparable papers and results: 
  * [ ] Results: https://paperswithcode.com/sota/token-classification-on-conll2003
  * [ ] Recent paper: https://arxiv.org/pdf/2406.17474v1
