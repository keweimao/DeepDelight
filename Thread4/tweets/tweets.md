# Broad Twitter Corpus
Introduced by Derczynski et al. in Broad Twitter Corpus: A Diverse Named Entity Recognition Resource : https://paperswithcode.com/paper/broad-twitter-corpus-a-diverse-named-entity

This paper introduces the Broad Twitter Corpus (BTC), which is not only significantly bigger, but sampled across different regions, temporal periods, and types of Twitter users. The gold-standard named entity annotations are made by a combination of NLP experts and crowd workers, which enables us to harness crowd recall while maintaining high quality. We also measure the entity drift observed in our dataset (i.e. how entity representation varies over time), and compare to newswire.

Link: https://paperswithcode.com/dataset/broad-twitter-corpus

Dataset: GateNLP/broad_twitter_corpus https://huggingface.co/datasets/GateNLP/broad_twitter_corpus

Dataset Card for broad_twitter_corpus


## Dataset Summary

This is the Broad Twitter corpus, a dataset of tweets collected over stratified times, places and social uses. The goal is to represent a broad range of activities, giving a dataset more representative of the language used in this hardest of social media formats to process. Further, the BTC is annotated for named entities.

See the paper, Broad Twitter Corpus: A Diverse Named Entity Recognition Resource, for details.


Supported Tasks and Leaderboards

## Named Entity Recognition
On PWC: Named Entity Recognition on Broad Twitter Corpus

Languages

English from UK, US, Australia, Canada, Ireland, New Zealand; bcp47:en


# Dataset Structure


## Data Instances

Feature	Count
Documents	9 551
Tokens	165 739
Person entities	5 271
Location entities	3 114
Organization entities	3 732

## Data Fields

Each tweet contains an ID, a list of tokens, and a list of NER tags

id: a string feature.
tokens: a list of strings
ner_tags: a list of class IDs (ints) representing the NER class

 0:  O 
 
 1: B-PER
 
 2: I-PER
 
 3: B-ORG
 
 4: I-ORG
 
 5: B-LOC
 
 6: I-LOC




## Data Splits


|Section |   Region   | Collection Period |         Description      |   Annotators   | Tweet Count |
| ------ | ---------- | ----------------- | ------------------------ | -------------- | ----------- |
|    A   |     UK     |           2012.01 | General collectionl      |      Expert    |        1000 |
|    B   |     UK     |        2012.01-02 | Non-directed tweets      |      Expert    |        2000 |
|    E   |    Global  |	          2014.07 | Related to MH17 disaste  | Crowd & expert |       	200 |
|    F	 | Stratified |	        2009-2014 |	Twitterati               | Crowd & expert |        2000 |
|    G	 | Stratified |	        2011-2014 |	Mainstream news          | Crowd & expert	|        2351 |
|    H	 |   Non-UK	  |              2014	| General collection	     | Crowd & expert	|        2000 |
 
The most varied parts of the BTC are sections F and H. However, each of the remaining four sections has some specific readily-identifiable bias. So, we propose that one uses half of section H for evaluation and leaves the other half in the training data. Section H should be partitioned in the order of the JSON-format lines. Note that the CoNLL-format data is readily reconstructible from the JSON format, which is the authoritative data format from which others are derived.

Test: Section F

Development: Section H (the paper says "second half of Section H" but ordinality could be ambiguous, so it all goes in. Bonne chance)

## Broad Twitter Corpus: A Diverse Named Entity Recognition Resource

COLING 2016  ·  Leon Derczynski, Kalina Bontcheva, Ian Roberts · https://paperswithcode.com/paper/broad-twitter-corpus-a-diverse-named-entity

One of the main obstacles, hampering method development and comparative evaluation of named entity recognition in social media, is the lack of a sizeable, diverse, high quality annotated corpus, analogous to the CoNLL{'}2003 news dataset. For instance, the biggest Ritter tweet corpus is only 45,000 tokens {--} a mere 15{\%} the size of CoNLL{'}2003. Another major shortcoming is the lack of temporal, geographic, and author diversity. This paper introduces the Broad Twitter Corpus (BTC), which is not only significantly bigger, but sampled across different regions, temporal periods, and types of Twitter users. The gold-standard named entity annotations are made by a combination of NLP experts and crowd workers, which enables us to harness crowd recall while maintaining high quality. We also measure the entity drift observed in our dataset (i.e. how entity representation varies over time), and compare to newswire. The corpus is released openly, including source text and intermediate annotations.

## Reference

Leon Derczynski, Kalina Bontcheva, and Ian Roberts. 2016. Broad twitter corpus:A diverse named entity recognition resource. InProceedings of COLING 2016,the 26th International Conference on Computational Linguistics: Technical Papers.1169–1179.
