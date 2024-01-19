
# Prompt Engineering for Precise Answers

## Prompt for Precise Answer in Falcon

### Prompt

Based on the following information only: 

"{Retrieved Chunks or Sentences}"

{QUESTION} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "{QUESTION}". 

### Result

Falcon responds with a very short answer, sometimes an exact match of the ground truth. 


## Example 1

Based on the following information only: 

"The teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years."

What was the amount of children murdered? Be precise in your answer and do NOT repeat any part of the question. 



Based on the following information only: 

"The teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years."

What was the amount of children murdered? Please provide the shortest answer and do NOT repeat any part of the question. 


### With Falcon

Based on the following information only: 

"The teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years."

What was the amount of children murdered? Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "What was the amount of children murdered?". 

Falcon: **19** victims.


Actual Answer: 19 


## Example 2

Based on the following information only: 

Moninder Singh Pandher was sentenced to death by a lower court in February.

When was Pandher sentenced to death? Be precise in your answer and do NOT repeat any part of the question. 


Actual Answer: February.



### Falcon


Based on the following information only: 

"Moninder Singh Pandher was sentenced to death by a lower court in February."

When was Pandher sentenced to death? Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "When was Pandher sentenced to death?". 

Falcon: **February**.

