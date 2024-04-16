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


### 5. Check Implementation (Fall 2023 Week 7)

1. One way to check and test (for a potential bug) is to **implement KL from scratch**: 
    * Instead of calling `F.kl_div()` in the `torch.nn` module
    * Implement our own `KLLoss()` based on simple revision (simplification) of `DLITELoss()`
    * The following is example code: 

Based on the sample zero-value masking, we can compute `KL` in the `forward` function: 

```python
def forward(self, logits, targets):
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)

        # One-hot encode the targets to get true probabilities
        true_probs = F.one_hot(targets, num_classes=probs.size(-1)).float()

        # Masks for non-zero elements of probs and true_probs
        mask_probs = probs > 0
        mask_true_probs = true_probs > 0

        # Calculate g function for non-zero elements using the mask
        kl_values = torch.zeros_like(probs)
        kl_values[mask_true_probs] = true_probs[mask_true_probs] * torch.log(true_probs[mask_true_probs]/probs[mask_true_probs])

        # Sum over all classes and average over the batch size
        loss = kl_values.sum(dim=-1).mean()

        return loss
```

The `KL` formula can be found at: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence, where $P$ is true probability distribution and $Q$ is the estimate (prediction). 

2. The result of the above `KL` implementation will confirm whether the probabilities have been used correctly: 
    * If this KL works well, the impelementation is good and the issue is with DLITE theory itself. 
    * If this doesn't, then review code and fix the issue until achieve the same good result with THIS `KL`. 
3. After the above correction, implement and test **TWO** additional methods below: 
    * $DLITE^{1/3}$, which is a metric distance. Use the final DLITE implementation, and compute its **Cube Root** as `loss`. 
    * **LIT** method, which is a sum of `g_values` **without** `delta_h_values`. 


### 6. Winter 2024 (Week 1)

1. Code debugging and testing: Is it producing correct results without 0 scores? 
2. Test NER predictions with CE, KL, DLITE loss functions. Report overall (weighted/micro/macro) precision, recall, F1 in one table: 

Example format: 
```csv
Loss    Train Iterations    Precision   Recall  F1
CE      100?                0.9         0.9     0.9
KL      100                 0.91        0.91    0.91
DLITE   ...
L1      ...
```

3. Consider additional loss functions such as: 
    * existing implementation of `nn.CosineEmbeddingLoss`, and/or 
    * Jessen-Shannon divergence: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence


References: 
* A good reference to additional loss functions: https://arxiv.org/pdf/2301.05579.pdf
* Loss functions in PyTorch: https://pytorch.org/docs/stable/nn.html#loss-functions

## Experiments

More datasets for NER: 
+ CoNLL: https://huggingface.co/datasets/conll2003
+ Broad Twitter Corpus: https://huggingface.co/datasets/strombergnlp/broad_twitter_corpus
+ NYTimes Annotated Corpus: https://catalog.ldc.upenn.edu/LDC2008T19

## References

* Comprehensive Overview of Named Entity Recognition: Models, Domain-Specific Applications and Challenges: https://arxiv.org/abs/2309.14084
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://aclanthology.org/N19-1423.pdf
* Exploring Cross-sentence Contexts for Named Entity Recognition with BERT: https://arxiv.org/abs/2006.01563
* NER-BERT: A Pre-trained Model for Low-Resource Entity Tagging: https://arxiv.org/abs/2112.00405

## DeBerta Reference

+ DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION: https://openreview.net/pdf?id=XPZIaotutsD
+ Github code for DeBERTA (Microsoft): https://github.com/microsoft/DeBERTa
+ DeBERTA-base fine-tuned NER: https://huggingface.co/geckos/deberta-base-fine-tuned-ner
+ DeBERTA-v3 large Conll2003 dataset: https://huggingface.co/tner/deberta-v3-large-conll2003 
+ KDDIE at SemEval-2022 Task 11: Using DeBERTa for Named Entity Recognition: https://aclanthology.org/2022.semeval-1.210.pdf
+ NER Papers with code Reference: https://paperswithcode.com/task/named-entity-recognition
