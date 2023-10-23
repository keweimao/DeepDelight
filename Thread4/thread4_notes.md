# Named Entity Recognition (NER)

## Focus

To identify named entities from local data via training and/or fine-tuning. 

## Data

Potential data to consider: 
+ A list of related datasets: https://github.com/juand-r/entity-recognition-datasets
+ NY Times Annotated corpus: https://catalog.ldc.upenn.edu/LDC2008T19
+ WikiANN: https://huggingface.co/datasets/wikiann
+ https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/data

## Systems

Potential systems to use or compare: 
+ spaCy: https://spacy.io/

## Models

Base model: 
1. Use a pre-trained model for word embedding, e.g. BERT, RoBERTa, or DistilBERT. 
2. Add classification NN on the model to predict named entities, e.g. person, organization, location, and O (outside)
3. Train and test. 

Loss functions: 
1. standard cross-entropy, KL
2. DLITE as Loss function

Related resources: 
+ BERT for NER: https://www.kaggle.com/code/thanish/bert-for-token-classification-ner-tutorial
+ Fine-tune BERT for NER: https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/


## Assignments

### 1. Exploration

+ Follow the tutorial to train/fine-tune BERT for NER. 
+ Test and report the results. 

### 2. Baseline (Fall 2023 Week 2)

* Reading to understand BERT: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
* Reading to understand BERT for classification: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
* Follow this to experiment and produce a baseline: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb

TODO: 
* Present major ideas from the above reading (understanding)
* How does it translate to BERT for token classification (NER)? 
* Produce NER baseline results in terms of Precision, Recall, F1, etc. (as in the tutorial). 

### 3. Training and Optimization (Fall 2023 Week 3)

* A Survey on Deep Learning for Named Entity Recognition: https://ieeexplore.ieee.org/abstract/document/9039685
* Fine-tune BERT for NER: https://skimai.com/how-to-fine-tune-bert-for-named-entity-recognition-ner/

TODO: 
* Investigate different parameters for training, e.g. learning rate, dropout, and experiment them to observe their impact on training efficient and result accuracy. 
* Investigate how to implement a custom LOSS function for NER training (on top of BERT). 
    * Also read DLITE paper and review DLITE_LOSS code from Thread #1

### 4. Training and Optimization (Fall 2023 Week 4)

Reading: 
* DLITE paper and the DLITE formula: https://arxiv.org/abs/2002.07888
* An example of DLITE Loss for `torch` in Thread #1: [dlite_loss_notes.md](../Thread1/dlite_loss_notes.md)

TODO: 
* Invesgate the potential to implement something similar with your NER implementation: 
    * Related implementation in Thread #1: [Jerry's Notebook](../Thread1/DLITE_CrossEntropy_Comparison_10_11_2023.ipynb)
    * First test something simple like `L1` Loss in your implementation
    * Then implement DLITE Loss after your custome `L1` works reliably


## Experiments


## References

* Comprehensive Overview of Named Entity Recognition: Models, Domain-Specific Applications and Challenges: https://arxiv.org/abs/2309.14084
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://aclanthology.org/N19-1423.pdf
* Exploring Cross-sentence Contexts for Named Entity Recognition with BERT: https://arxiv.org/abs/2006.01563
* NER-BERT: A Pre-trained Model for Low-Resource Entity Tagging: https://arxiv.org/abs/2112.00405

