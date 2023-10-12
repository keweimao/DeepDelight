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
    * Pay attention to the architecture and prompt/context template
    * Chunking and embedding

TODO: 
1. Replicate the implementation of the above, with the following in mind: 
    * Text document chunking, embdedding to vector stores, 
    * Try if you can use GPT4All's embedding, instead Google Vertex AI (in the video)
2. Explore and investigate potential benchmark dataset to use. 
3. Think about the type of questions and answers to focus on. 
    * For example, tabular data with implicit inforamtion that require logical understanding of "ranking" and "aggregation" is a challenging task. 
    * Do we want to take up this special data/question type and come up with an innovative solution, e.g. with ways to **encode** ranking and aggregation in the data with proper **chunking**? 

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