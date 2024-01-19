# GPT+IR

## Purpose

To combine the power of GPT with a retrieval engine for accurate information retrieval and question answering. 

## Focus

GPT+IR for question answering. 

## Data

Potential dataset to use: 
+ Stanford QA dataset: https://rajpurkar.github.io/SQuAD-explorer/


## TODOs

### 1. Exploration

+ Explore LangChain w/ GPT4ALL (or any ready to use GPT models)
+ Try the above for QAs: https://python.langchain.com/docs/use_cases/question_answering/
+ Use a small collection of documents to be vectorized/stored, e.g. Wikipedia docs about Fifa world cups
+ Create a couple of questions related, e.g. Which team won the 2022 world cup? 
+ Test the questions and report what you find in the result/answers. 

### 9/1/2023 Notes: 

* Observation: 
    * Answer accuracy varies
    * Long response time to each question
* TDOD: 
    * Split the long FIFA document into multiple documents manually (small and structured/specific for certain questions). 
    * Load documents from within a directory. 
    * Test the same questions again. 

### 1. Baseline Implementation and Testing (Fall 2023 Week 3)

* Generative AI / LLM - Document Retrieval and Question Answering: youtube.com/watch?v=inAY6M6UUkk
    * Consider the architecture and prompt/context template
    * Consider chunk and chunk overlap sizes for embedding
        * See chunking parameters at 6:15: https://youtu.be/inAY6M6UUkk?si=A62nOAaPpE1G0CkP&t=375
* Consider chunking strategies: 
    * Fixed chunking with fixed overlap, enough? 
    * Semantic chunking based on text structures, e.g. sentences or paragraphs
    * Adaptive or hierarchical chunking: start with larger chunks, and break down into smaller ones if no good (confident) answer is identified. 
        * How do we dynamically evaluate confidence of an answer? 

TODO: 
1. Review the implementation of the above, with the following in mind: 
    * Text document **chunking**, **overlap**, before embdedding to vector stores, 
    * Investigate which model of GPT4All's embedding is most effective? 
    * Test it in your implementation
2. Explore and investigate potential benchmark dataset to use. 
3. Think about the type of questions and answers to focus on. 
    * For example, tabular data with implicit inforamtion that require logical understanding of "ranking" and "aggregation" is a challenging task. 
    * Do we want to take up this special data/question type and come up with an innovative solution, e.g. with ways to **encode** ranking and aggregation in the data with proper **chunking**? 

### 2. Further Reading and Ideation (Fall 2023 Week 4)

Reading: 
* Retrieval Augmented Generation (RAG): https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590/Improving-the-Domain-Adaptation-of-Retrieval

Benchmark datasets: 
* QAConv: https://github.com/salesforce/QAConv
* NewsQA: https://github.com/Maluuba/newsqa

TODO: 
1. Explore the QAConv and NewsQA datasets or other that have been used in related research. Pick one to test. 
2. Experiment on the following chunking strategies and observe performances: 
    * Fixed chunking: 200, 400, 800, 1600; combined with 0%, 10%, 20% and 30% overlap. (16 combinations)
    * Semantic structure chunking: sentence, paragraphs, and sections. (No overlap)
    * Record performance for each configuration, e.g. EM (exact match) and F1

## 3. Experimentation (Fall 2023 Week 7)

1. Optimization: 
    * Looks like the gpt4all module is already optimized GPUs (Apple Metal)
    * Text splitter and embedder instances can be called ONCE outside loops
2. Improvement: 
    * Change `1similar_search` to `similarity_search_by_vector()` with query embedding (DONE)
    * Consider `InstructEmbedding` for search queries (questions), along with other models (TODO)
3. Current results on NewsQA:
    * Tested chunk_sizes = [25, 50, 75, 100, 125, 150, 175, 200, 400, 800, 1600]
    * Best on first 100 stories at around chunk_size=200 with 0 overlap, low F1<0.1 and 0 EM
    * Inconsistent F1 scores in different runs, please check F1 calculation.
    * Use the 1st story and all 9 questions, print ground truth, predicted answer, and F1 for checking.
4. Mengyang to take on QAConv dataset?
    * First use a TINY subset of: https://github.com/Maluuba/newsqa
    * Implement and test implementation on the TINY data. 

## 4. Improvement (Winter 2024 Week 1)

For Lixiao on `NewsQA` and Mengyang on `QAConv` dataset respectively, follow the same procedure: 

1. Use the GPT4All Falcon (8GB memory use) model for the final answer generation. 
2. Test the following prompt (see prompts.md for examples) and evaluate precision, recall, and F1: 

```json
Based on the following information only: 

"{Retrieved Chunks or Sentences}"

{QUESTION} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "{QUESTION}". 
```

3. Test the above on a small subset only, e.g. with 1 story and/or 10 questions or so. 
4. If the above yield decent results, create code as `.py` for experiments on the full dataset. 
5. NEXT: Will go back to different chunks and overlaps for (baseline) comparision, using the same prompt. 
6. NEXT: Consider other even smaller models for answer generation, i.e. less memory and faster responses. 


## Additional Readings and Resources

### A. General Articles and Resources

1. IBM Research Blog on RAG as an AI framework that improves the quality of LLM-generated responses by grounding the model on external knowledge sources​: https://research.ibm.com/blog/retrieval-augmented-generation-RAG
2. Hugging Face brief documentation on RAG: https://huggingface.co/docs/transformers/model_doc/rag
3. Key concepts in RAG: https://www.datastax.com/guides/what-is-retrieval-augmented-generation
4. Azure ML on RAG: https://learn.microsoft.com/en-us/azure/machine-learning/concept-retrieval-augmented-generation?view=azureml-api-2

### B. Research Papers

1. ArXiv paper with a proposed retrieval-augmented generative QA model for **event argument extraction**, focusing on the effectiveness of using RAG in various settings​: https://arxiv.org/abs/2211.07067
2. Meta Research paper with proposed RAG: https://arxiv.org/abs/2005.11401, also discussed briefly at: https://www.promptingguide.ai/techniques/rag
3. A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT: https://arxiv.org/abs/2302.11382

### C. Notebooks for Ideas

1. Amazon SageMaker JumpStart provides sample notebooks demonstrating question answering tasks using a RAG-based approach with large language models (LLMs), focusing on domain-specific text outputs: https://aws.amazon.com/blogs/machine-learning/question-answering-using-retrieval-augmented-generation-with-foundation-models-in-amazon-sagemaker-jumpstart/
2. RAG Evaluation: https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a

### Data

TREC QA data: https://trec.nist.gov/data/qa/t2007_qadata.html

### Related Systems and Implementations

+ LangChain: https://python.langchain.com/docs/get_started/introduction.html
    + LangChain for QAs: https://python.langchain.com/docs/use_cases/question_answering/
+ GPT4ALL: https://gpt4all.io/index.html
    + github: https://github.com/nomic-ai/gpt4all
    + for langchain: https://python.langchain.com/docs/integrations/llms/gpt4all.html
    + data like: https://github.com/nomic-ai/gpt4all-datalake

### Reading and References

* REALM: Retrieval-Augmented Language Model Pre-Training: https://arxiv.org/abs/2002.08909
* In-Context Retrieval-Augmented Language Models: https://arxiv.org/abs/2302.00083
* Standford & Adobe Research (2023). Question Answering over Long, Structured Documents: https://arxiv.org/abs/2309.08872
* UCSB & Google Research (2021). Open Question Answering over Tables and Text: https://arxiv.org/abs/2010.10439
* Google (2020). Retrieval Augmented Language Model Pre-Training: http://proceedings.mlr.press/v119/guu20a.html?ref=https://githubhelp.com