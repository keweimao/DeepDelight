# Running Scripts Index
This file is used for maintaining different version of running code and document changes and updates to prevent confusions between different experiments. The running scripts here only include NewQA dataset scripts only.

## Version 1

[1. newsqa_rag_script_lyang.py](https://github.com/keweimao/DeepDelight/blob/main/Thread2/Running_Scripts/1.%20newsqa_rag_script_lyang.py)  \
First version of full dataset running loop, include:
- Updated prompt from previous results
- CSV formatted output file

## Version 2

[2_newsqa_rag_script_lyang_output_fix.py](https://github.com/keweimao/DeepDelight/blob/main/Thread2/Running_Scripts/2_newsqa_rag_script_lyang_output_fix.py)  \
Differentce compared to previous version:
- Add chunk size and overlap size into CSV output
- Add log generation

[2_newsqa_rag_script_lyang_with_normalization.py](https://github.com/keweimao/DeepDelight/blob/main/Thread2/Running_Scripts/2_newsqa_rag_script_lyang_with_normalization.py)  \
Differentce compared to previous version:
- Add chunk size and overlap size into CSV output
- Add log file generation
- **Add normalization and stemming before comparing predicted and actual answers**
- **Fix normalized sentence output newline issue [1/26/2024]**

[2_2_newsqa_rag_script_lyang_with_sentence_and_dist.py](https://github.com/lixiao-yang/DeepDelight/blob/main/Thread2/Running_Scripts/2_2_newsqa_rag_script_lyang_with_sentence_and_dist.py)
Differentce compared to previous version:
- Add chunk size and overlap size into CSV output
- Add log file generation
- Add normalization and stemming before comparing predicted and actual answers
- Fix normalized sentence output newline issue [1/26/2024]
- **Add sentence-based structure and with adjustable parameter `top_n_sentences`**
- **Add parameter `dist_functions` to enable cosine/pairwise distance calculations**

[2_3_newsqa_rag_script_lyang_sentence_revised.py](https://github.com/lixiao-yang/DeepDelight/blob/main/Thread2/Running_Scripts/2_3_newsqa_rag_script_lyang_sentence_revised.py)
Differentce compared to previous version:
- ~~Add chunk size and overlap size into CSV output~~
- **Modified log and csv file output [1/30/2024]**
- Add normalization and stemming before comparing predicted and actual answers
- Fix normalized sentence output newline issue [1/26/2024]
- Add sentence-based structure and with adjustable parameter `top_n_sentences`
- Add parameter `dist_functions` to enable cosine/pairwise distance calculations
- **Incorporate `top_nsentences` and `dist_functions` as lists of parameters to enable loop**

[2_4_newsqa_rag_script_lyang_whole_story.py](https://github.com/lixiao-yang/DeepDelight/blob/main/Thread2/Running_Scripts/2_4_newsqa_rag_script_lyang_whole_story.py)
Differentce compared to previous version:
- ~~Add chunk size and overlap size into CSV output~~
- Add log file generation
- Add normalization and stemming before comparing predicted and actual answers
- Add sentence-based structure and with adjustable parameter `top_n_sentences`
- Add parameter `dist_functions` to enable cosine/pairwise distance calculations
- **Whole story as the prompt**
