# Schedule, Modules, and Research Threads

This document outlines the (roughly) weekly modules and topics as part of the research. 
They may include readings, tutorials, exercises, as well as assignments for development (implementation), experiments (tests), and reporting (writing).

## Research Threads and Objectives

Current Threads (Summary 2023): 

1. Transformer with DLITE for predictive analytics (Sonia, Michael)
   * With applications in summarization and/or named entity recognition
2. GPT fine-tuning for information/fact/answer retrieval (Jerry, Lixiao)

Potential Future Projects: 

* A distributed network of personal agents based on GPTs

## Text and Readings

* Some readings will come from a book draft in `Reading/Ke_Book_draft.pdf`
* Others will include open book chapters, online tutorials, and video demos

## Module 1: Overview and Introduction

### Reading: 

1. `Reading/Ke_Book_draft.pdf`: 
   1. Chapter `Text and Human Language` (p.107-)
   2. Chapter `Search and Retrieval` (p.145-)
2. Language Models for IR:
  + https://nlp.stanford.edu/IR-book/html/htmledition/language-models-for-information-retrieval-1.html
  + https://www.elastic.co/blog/language-models-in-elasticsearch

### Assignment 

on Simple Language Models: 

1. Recreate the two language models (uni-gram and bi-gram) on the Tiny Shakespeare text data in the `Shake_LM` folder. 
2. Optional: Compile a collection of your own writings or articles written by one author, and test the models. 
3. Write a brief report within one page only: 
   1. Compare the two models, any differences or similarities? 
   2. Your thoughts on the results and how they can be improved. 

## Module 2: Neural Networks and Reinforcement Learning

### Reading

1. State of GPT: https://youtu.be/bZQun8Y4L2A
   * Be mindful of GPT training pipeline, e.g. at 1:00. 
2. Chapter `Binary Questions` (p.81-) in `Reading/Ke_Book_draft.pdf` 
   * Video on non-linear classification (NN): https://www.youtube.com/watch?v=P1RAjm6di20
3. Weimao Ke's Tutorial on Reinforcement Learning: http://keensee.com/pdp/research/rl_taxi.html
   * Video on the above tutorial: https://youtu.be/QUPpKgXJd5M

### Exercise

Pick one of the following for the exercise: 

1. Build a Neural Network: https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
2. Reinforcement Learning with Gym: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

## Module 3: Transformer & GPT

### Reading & Tutorial

1. Transformer explained (on Attention is All You Need paper): 
https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634
2. Challenges and opportunities with LLMs: https://towardsdatascience.com/four-llm-trends-since-chatgpt-and-their-implications-for-ai-builders-a140329fc0d2

### Assignment 

1. Follow this video tutorial `Andrej Karpathy's Let's Build GPT`: https://www.youtube.com/watch?v=kCc8FmEb1nY
2. Build the character-based uni-gram model on Tiny Shakespeare data. 
3. Train and test it using a **small** number of parameters, unless you have a powerful GPU machine. :)


## Module 4: Information Gain and Loss Functions

### Reading

1. Chapter `Information in the Chaos` (p.61-) in `Reading/Ke_Book_draft.pdf` 
2. DLITE for Big Data and IR: https://ieeexplore.ieee.org/document/10020937
2. Video presentation of DLITE: https://www.youtube.com/watch?v=qQXCgmX8sOk

### Exercise

1. Implement the DLITE function in Python (e.g. with Pandas or PyTorch). 
2. Parameters should be any two discrete probability distributions $P$ and $Q$. 
3. Test combinations of P and Q to show they produce correct scores. 

## Module 5: Pretrain, Finetune, or +Local Pretrain

+ Pretrained model finetuning? 
+ Vs. Pretrained model PLUS locally pretrained (mini) model
+ How do they collaborate for the best performance? 
+ What is most cost-effective, global retrain, local pretrain? 


+ GPT Alignment: https://openai.com/research/instruction-following
+ Learning from HP: https://openai.com/research/learning-from-human-preferences


## Module 6: LLM plus Retrieval

+ Augmented language models: https://bdtechtalks.com/2023/04/03/augmented-language-models/
+ REALM: https://ai.googleblog.com/2020/08/realm-integrating-retrieval-into.html
+ In context REALM: https://bdtechtalks.com/2023/01/30/ai21labs-llms-ralm/


## Potential Leads

GPT for private data retrieval: 
1. LangChain for loading personal data into ChatGPT for retrieval: 
2. Or LLaMa_index: https://github.com/jerryjliu/llama_index
3. A simple application here: https://www.youtube.com/watch?v=9AXP7tCI9PI&t=29s


## Additional Reading

* Simplified explanation of GPT: https://arstechnica.com/science/2023/07/a-jargon-free-explanation-of-how-ai-large-language-models-work/


