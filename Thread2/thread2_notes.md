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