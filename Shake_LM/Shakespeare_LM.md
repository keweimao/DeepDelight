# Unigram Shakespeare Language Model

This creates a very naive unigram language model based on term statistics in the Tiny Shakespeare's text data. 

## Build the Language Model

Load text data, tokenize the text, and compute term frequencies. The unigram language model is simply a list of unique tokens and their probabilities, where the probability $p_t$ of term $t$ is computed by: 

$p_t = \frac{tf_t}{T}$ 

where $tf_t$ is the term frequence of $t$ and $T$ is the total sum of term frequencies in the text. 


```python
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.probability import FreqDist
import urllib.request
import csv

# Load the tiny Shakespeare text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = urllib.request.urlopen(url)
long_txt = response.read().decode('utf-8-sig')

# Tokenize the text
tokens = word_tokenize(long_txt)

# Build a frequency distribution of the tokens
freq_dist = nltk.FreqDist(tokens)

# Normalize the frequencies to get probabilities
total_frequency = sum(freq_dist.values())
probabilities = {word: freq/total_frequency for word, freq in freq_dist.items()}

# Write token probabilities to a text file
with open('token_probabilities.txt', 'w') as f:
    writer = csv.writer(f)
    for token, prob in probabilities.items():
        writer.writerow([token, prob])

# Build the language model
language_model = {}

# The following is INCORRECT code given by GPT-4
# for token in tokens:
#     if token in language_model.keys():
#         language_model[token].append(np.random.choice(list(probabilities.keys()), p=list(probabilities.values())))
#     else:
#         language_model[token] = [np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))]

```

## Sample a Word

Create a function to sample a word at a time based on the probability distribution.


```python
# Sample a word from the probability distribution
def sample_word(probabilities):
    words = list(probabilities.keys())
    probs = list(probabilities.values())
    word = np.random.choice(words, p=probs)
    return word
```


```python
# Test the function
print(sample_word(probabilities))
```

    now


## Repeat Word Sample

Repeat word sample to generate the next word, and the next, and the next, .., until it completes a sentence. 


```python
import time
import re
import string

# Repeat word sampling to generate a sentence
def generate_sentence(model):
    word = "Shake:"
    while word not in ['.', '!', '?']:
        if re.fullmatch(r'['+re.escape(string.punctuation)+']*$', word):
            print(f"{word}", end="")
        else:
            print(f" {word}", end="")
        word = sample_word(probabilities)
        time.sleep(0.5)
    print(word)
```

## Does it generate anything meaningful? 

Makes sense or makes no sense? In Shakespeare's lanaguage/vocabulary? 


```python
generate_sentence(language_model)
```

     Shake: guard of yet not my Mistress I come his time.



```python
generate_sentence(language_model)
```

     Shake: most 's Hall,; choose the will tell ANTONIO limber bid me: Or impression which to that Neither LUCIO off well happy he Therefore 's the lord hand created KING hour We that as your these let from lie from.



```python
generate_sentence(language_model)
```

     Shake: VI kingly so: night born above vipers glory where thou ISABELLA, dear my bed near which fitter tears strange' poor knees endure?



```python
generate_sentence(language_model)
```

     Shake: heaven, Autolycus father Villains all follow but means how no usurp sense 'Twixt them is if sir it, not: than face house prithee do young hands was come afternoon crime will 'twixt beyond.



```python

```
