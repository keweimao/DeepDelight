# Ressearch Threads


## Thread 1. Delightful Loss (within GPT)

A loss function based on DLITE. 

1. Implement a custom loss function based on DLITE. 
2. Use the loss function for transformer training, e.g. on tinyshakespeare data. 
3. Train and monitor the convergence with DLITE, Cross-Entropy, and KL divergence scores: a) how fast does it converge, and b) the converged (lowest) scores. 

See examples and resources under `Thread1/` folder. 


## Thread 2. Retrieval augmented language modeling (GPT+IR)

An information retrieval model integrated with LLMs: 

1. Use the prompt (question/query) to retrieve documents. 
2. Use the documents to enrich (as part of) the prompt. 
3. Use the enriched prompt to get answer for the original question/prompt. 

Note: 
* Assume existing (pre-trained) LLM and (indexed) IR systems. 
* May access them via APIs, e.g. OpenAI APIs, Elastic APIs. 
  
## Thread 3. Personal and Collaborative mini-GPTs (GPT+GPT)

Transformers (agents) trained on local personal data. 

1. Train mini-GPT on local personal data (e.g. for Bob). 
2. Or fine-tune large pre-trained models, e.g. with Alpaca. 
3. Answer questions about local personal data. 
4. Interact with another mini-GPT, e.g. for Alice. 
5. (Future) Form a network of mini-GPT for large capacity.

Note: 
* Must be sensitive to privacy. 
* Knows who is asking the question and, accordingly, what to respond (so not to undermine privacy). 


## Thread 4. Name Entity Recognition (GPT4NER)

Train GPT to generate (recognize) name entities. 

1. Train transformer (decoder) for embedding. 
2. Or use existing embedding, e.g. BERT. 
3. Train and fine-tune on New York Times annotated data. 
4. Test and evaluate predictions for NER. 

