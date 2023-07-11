
# Summer 2023 Research Notes

## Week 1

Reading from Module 4: 
https://towardsdatascience.com/four-llm-trends-since-chatgpt-and-their-implications-for-ai-builders-a140329fc0d2

### Big picture on the `transformation` technology for AI: 

* Training objectives
  1. Autoregression (generative): common focus, not analytical
  2. Autoencoding (extraction/analytics)
  3. Sequence-to-sequence (translation)
* Size, efficiency, and scalability
* Data privacy, personalization, and user experience
  * Personalized GPTs (agents) trained (or fine-tuned) on private data
  * Individuals have control over their models/agents
  * Distributed agents capable of socializing and working with each other
    * Potential for scalability, task diversity, privacy, personalization, etc.
* GPTs for information/answer/fact retrieval
  * GPT being `generative`: problem of hallucination
  * GPT being `pre-trained`: not capable of `knowing` new data
  * To integrate with IR to retrieve data
  * Get trained/tuned on NEW data
  * Get reinforced (RL) to provide `factual` answers, something that can be referenced

And then there are: 
* Objective/gain/loss functions for LLM training
* Cross-entropy and KL divergence are very common
* Ke's DLITE, Discounted Least Information Theory for Entropy (Ke 2020 & 2022)

### B. Practical: 
* LLM-based frameworks such as [LangChain](https://github.com/hwchase17/langchain), [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) and [LlamaIndex](https://github.com/jerryjliu/llama_index)
  * Allows integration with private data and its index
* How about integration with an IR engine, e.g. Elastic? 
* Distributed models with web APIs

### C. Research Objectives: 

1. Improved autoencoding for analytical tasks such as summarization, extraction, and named entity detection, e.g. with DLITE. 
1. Personalized GPTs for local information retrieval and question answering, e.g. Elastic and RL. 
2. Distributed GPTs for collaborative information retrieval and analytical tasks. 


## Week 2

### Week 1 Review

+ Language model exercise & presentation
+ Thoughts and comments
  + Potentials and limits
  + Uni-gram to bi-gram: size of context
+ Questions on reading? 


Notes: 
1. bag of words, but not exactly? 
2. what if more context? 
3. how long does it run, e.g. uni- and bi-gram? 
4. tokens as words? 

### Week 2 Notes

+ Overview of GPT training, pipeline, and "tricks"
+ Reading on ML and predictive analytics
+ Linear to non-linear models (e.g. NN)
+ Reinforcement learning (e.g. Q-learning)
+ Exercise, questions, and thoughts for next week



