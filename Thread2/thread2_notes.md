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


## Additional Resources

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