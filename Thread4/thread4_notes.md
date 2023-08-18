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


## TODOs

### 1. Exploration

+ Follow the tutorial to train/fine-tune BERT for NER. 
+ Test and report the results. 



## Experiments


## References

* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://aclanthology.org/N19-1423.pdf
* Exploring Cross-sentence Contexts for Named Entity Recognition with BERT: https://arxiv.org/abs/2006.01563

