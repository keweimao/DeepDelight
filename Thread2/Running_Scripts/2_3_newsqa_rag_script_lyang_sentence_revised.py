"""
---------Parameters to be changed between different devices----------
1. Models and directory
2. Input and output directory
3. Device selection (CPU/GPU)
4. Libraries configuration set adjustments
"""


import logging
import time
from collections import Counter
from collections import defaultdict
import csv
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from heapq import nlargest
import torch.nn as nn
# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# import numpy as np

########## Config for Dr.Ke ############
# from langchain_community.llms import GPT4All
# from pathlib import Path
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
# from transformers import set_seed
# from langchain_community.embeddings import GPT4AllEmbeddings

######### Config for Lixiao ############
from langchain.llms import GPT4All
from pathlib import Path
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from transformers import set_seed
from langchain.embeddings import GPT4AllEmbeddings

# Function to normalize and stem text
def normalize_and_stem(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())  # Normalize to lowercase and tokenize
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]  # Stemming and removing punctuation
    return ' '.join(stemmed_tokens)

# Modified function to calculate Exact Match (EM) score
def calculate_em(predicted, actual):
    return int(predicted == actual)

# Modified function to calculate the token-wise F1 score and return precision and recall
def calculate_token_f1(predicted, actual):
    predicted_tokens = predicted.split()
    actual_tokens = actual.split()
    common_tokens = Counter(predicted_tokens) & Counter(actual_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0, 0, 0  # Return zero precision, recall, and F1 score

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(actual_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall

def newsqa_loop(data, llm, output_csv_path, output_log_path, max_stories,
                top_n_sentences, dist_functions, instruct_embedding_model_name, instruct_embedding_model_kwargs, 
                instruct_embedding_encode_kwargs, QA_CHAIN_PROMPT):
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Chunk_size', 'Chunk_Overlap', 'Time', 'Story Number', 'Question Number', 'EM', 'Precision', 'Recall', 'F1'])

        # Embedding for story sentences
        hf_story_embs = HuggingFaceInstructEmbeddings(
            model_name=instruct_embedding_model_name,
            model_kwargs=instruct_embedding_model_kwargs,
            encode_kwargs=instruct_embedding_encode_kwargs,
            embed_instruction="Use the following pieces of context to answer the question at the end:"
        )

        # Embedding for questions
        hf_query_embs = HuggingFaceInstructEmbeddings(
            model_name=instruct_embedding_model_name,
            model_kwargs=instruct_embedding_model_kwargs,
            encode_kwargs=instruct_embedding_encode_kwargs,
            query_instruction="How does this information relate to the question?"
        )
        
        start_time = time.time()

        for dist_function in dist_functions:
            print(f"\n{time.time()-start_time} Processing distance function: {dist_function}")
            last_time = time.time()
            
            for top_n_sentence in top_n_sentences:
                print(f"\n{time.time()-start_time}\tTop sentences number: [{top_n_sentence}]")
        
                for i, story in enumerate(data['data']):
                    if i >= max_stories:
                        break

                    now_time = time.time()
                    print(f"\n{now_time - start_time}\t{now_time - last_time}\t\tstory {i + 1}: ", end='')
                    last_time = now_time
                    
                    text_splitter = RecursiveCharacterTextSplitter()

                    # Segment story text into sentences and embed each sentence
                    sentences = sent_tokenize(story['text'])
                    sentence_embs = hf_story_embs.embed_documents(sentences)

                    # Text splitting for chunk-based processing (if required)
                    all_splits = text_splitter.split_text(story['text'])
                    
                    # Pass the embedding model to Chroma.from_texts
                    vectorstore = Chroma.from_texts(texts=all_splits, embedding=hf_story_embs)

                    # Initialize the QA chain with the vectorstore as the retriever
                    qa_chain = RetrievalQA.from_chain_type(
                        llm, 
                        retriever=vectorstore.as_retriever(), 
                        chain_type="stuff",
                        verbose=False,
                        chain_type_kwargs={
                            "prompt": QA_CHAIN_PROMPT,
                            "verbose": False},
                        return_source_documents=False
                    )
                        
                    for j, question_data in enumerate(story['questions']):
                        if question_data['isAnswerAbsent']:
                            continue  # Skip this question because an answer is absent

                        question = question_data['q']
                        question_emb = hf_query_embs.embed_documents([question])[0]

                        # Compute cosine similarity scores with each sentence
                        if dist_functions == 'pairwise':
                            scores = [pdist(torch.tensor(sentence_emb).unsqueeze(0), torch.tensor(question_emb).unsqueeze(0)).item() for sentence_emb in sentence_embs]
                        else:
                            scores = [torch.cosine_similarity(torch.tensor(sentence_emb).unsqueeze(0), torch.tensor(question_emb).unsqueeze(0))[0].item() for sentence_emb in sentence_embs]

                        # Find the sentences with the top x highest scores
                        top_scores_indices = nlargest(top_n_sentence, range(len(scores)), key=lambda idx: scores[idx])
                        context_for_qa = " ".join([sentences[idx] for idx in top_scores_indices])

                        # Check if there is a consensus answer and extract it
                        consensus = question_data['consensus']
                        if 's' in consensus and 'e' in consensus:
                            actual_answer = story['text'][consensus['s']:consensus['e']]
                        else:
                            continue  # No consensus answer, skip to the next question

                        # Get the prediction from the model
                        result = qa_chain({"context": context_for_qa, "query": question})
                        # print(context_for_qa)
                        
                        # Extract and process the predicted answer
                        predicted_answer = result['result'] if isinstance(result['result'], str) else ""
                        
                        # Normalize and stem the predicted and actual answers
                        normalized_predicted_answer = normalize_and_stem(predicted_answer)
                        normalized_actual_answer = normalize_and_stem(actual_answer)

                        # Calculate the F1 score, precision, and recall using normalized and stemmed answers
                        f1_score_value, precision, recall = calculate_token_f1(normalized_predicted_answer, normalized_actual_answer)
                        em_score = calculate_em(normalized_predicted_answer, normalized_actual_answer)

                        # Write the scores to the file
                        writer.writerow([dist_function, top_n_sentences, time.time() - start_time, i, j, em_score, precision, recall, f1_score_value])
                        
                        with open(output_log_path, 'a') as details_file:
                            details_file.write(f"Distance Function: {dist_function}\n")
                            details_file.write(f"Top sentences retrieved: {top_n_sentence}\n")
                            details_file.write(f"Story: {i}\n")
                            details_file.write(f"Question: {j}\n")
                            details_file.write(f"Correct Answer: {actual_answer}\n")
                            details_file.write(f"Normalized Actual Answer: {normalized_actual_answer}\n")
                            details_file.write(f"Predicted Answer: {predicted_answer}\n")
                            details_file.write(f"Normalized Predicted Answer: {normalized_predicted_answer}\n")
                            details_file.write(f"Time: {time.time() - start_time}\n")
                            details_file.write(f"EM Score: {em_score}\n")
                            details_file.write(f"Precision: {precision}\n")
                            details_file.write(f"Recall: {recall}\n")
                            details_file.write(f"F1: {f1_score_value}\n")
                            details_file.write("----------------------------------------\n")

                    # Cleanup
                    del qa_chain
                    del vectorstore
                    del all_splits

                # End of the story loop
                del text_splitter

############## Running Parameters ##############
max_stories = 100
random_seed = 123
top_n_sentences = [1, 2] # Use top n scored sentence as embedding
dist_functions = ['pairwise', 'cosine'] # Default value is cosine similarity
model_location = "C:/Users/24075/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"
# model_location = "/Users/wk77/Library/CloudStorage/OneDrive-DrexelUniversity/Documents/data/gpt4all/models/gpt4all-falcon-q4_0.gguf"
# model_location = "/Users/wk77/Documents/data/gpt4all-falcon-newbpe-q4_0.gguf"
# model_location = "/Users/wk77/Documents/data/mistral-7b-instruct-v0.1.Q4_0.gguf"
input_file_path='C:/NewsQA/combined-newsqa-data-story2.json'
# input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/newsqa-data-v1.csv"
# input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined-newsqa-data-v1.json"
# input_file_path = "/Users/wk77/Documents/git/DeepDelight/Thread2/data/combined-newsqa-data-story1.json"
output_csv_path = '../results/story2_score_test7.csv'
# output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/story1_scores_test.csv"
# output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined_scores_test.csv"
output_log_path = '../results/story2_score_test7.log'

# Initialize PairwiseDistance
pdist = nn.PairwiseDistance(p=2.0, eps=1e-06)

##################################################
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)  # This will show only warnings and errors
logging.basicConfig(level=logging.ERROR)

print("Loading data.")
data = json.loads(Path(input_file_path).read_text())

print("Setting template.")
template_original = """
                    Based on the following information only: 
                    
                    {context}
                    
                    {question} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "{question}".

                    Answer:
                    """
QA_CHAIN_PROMPT_ORIGINAL = PromptTemplate.from_template(template_original)

print("Random seeding.")
set_seed(random_seed)

# Results storage
f1_results = defaultdict(list)
em_results = defaultdict(list)
text_results = []

# Initialize the language model and the QA chain
print("Loading LLM.")
llm = GPT4All(model=model_location, max_tokens=2048, seed=random_seed)

print("Preparing Parameters.")
# HuggingFace Instruct Embeddings parameters
instruct_embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
instruct_embedding_model_kwargs = {'device': 'cpu'}
# instruct_embedding_model_kwargs = {'device': 'mps'}
instruct_embedding_encode_kwargs = {'normalize_embeddings': True}

# The following code would iterate over the stories and questions to calculate the scores
start_time = time.time()
print(f"{start_time} Started.")                

# Main Function Execution
print("Processing.")
newsqa_loop(data, llm, output_csv_path, output_log_path, max_stories, top_n_sentences, dist_functions, instruct_embedding_model_name,
            instruct_embedding_model_kwargs, instruct_embedding_encode_kwargs, QA_CHAIN_PROMPT_ORIGINAL)