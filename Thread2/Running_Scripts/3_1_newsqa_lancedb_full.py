"""
---------Parameters to be changed between different devices----------
1. Models and directory
2. Input and output directory
3. Device selection (CPU/GPU) - LINE 46 AND 300
4. Install tantivy package for full index search

Reference:
1. similarity_search_by_vector(): https://python.langchain.com/docs/modules/data_connection/vectorstores/
2. LanceDB Code documentation Q&A bot example with LangChain: https://lancedb.github.io/lancedb/notebooks/code_qa_bot/
3. LanceDB embedding functions: https://lancedb.github.io/lancedb/embeddings/embedding_functions/
4. LanceDB available models: https://lancedb.github.io/lancedb/embeddings/default_embedding_functions/#sentence-transformers
5. https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/examples/Code-Documentation-QA-Bot/main.ipynb
6. Query via text: https://lancedb.github.io/lancedb/notebooks/DisappearingEmbeddingFunction/#querying-via-text
7. Pre and post-filtering: https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering
8. Create FTS index on single column: https://lancedb.github.io/lancedb/fts/#create-fts-index-on-single-column
9. LanceFtsQueryBuilder: https://lancedb.github.io/lancedb/python/python/#lancedb.query.LanceFtsQueryBuilder
remove chunk size and overlap size for

add similarity func, embedding_func into logging 
code structure adjustment, incorporate prompt parameter
"""
# Make S
# pip install tantivy==0.20.1

import os
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
from pathlib import Path
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import LanceDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from transformers import set_seed
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, vector
# from langchain_community.vectorstores import LanceDB

print("LanceDB config.")
registry = EmbeddingFunctionRegistry.get_instance()
func = registry.get("sentence-transformers").create(device="cpu")  

def process_and_store_chunks(data, db_dir, embedding_model, chunk_size, overlap_percentage, text_splitter, embedding_func):
    story_id = data['storyId']
    story_text = data['text']
    print("\n In process function")
    
    # Ensure the database directory exists
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    # Define the unique table name for this story's chunks
    adjusted_story_id = story_id[14:-6] if len(story_id) > 6 else story_id
    table_name = f"{adjusted_story_id}-chunks"
    story_db_dir = db_dir # os.path.join(db_dir, table_name)
    print(db_dir)

    # Connect to or create the database
    db = lancedb.connect(story_db_dir)

    try:
        table = db.open_table(table_name)
        print("\n Found existing table in the location.")
    except Exception as e:
        chunk_splits = text_splitter.split_text(story_text)
        chunk_embs = embedding_model.embed_documents(chunk_splits)
        
        # Prepare the data for storage
        chunks_embs_data = [
            {
                "storyId": adjusted_story_id,
                "chunk_size": chunk_size,
                "overlap_size": overlap_percentage,
                "embedding_function": embedding_func,
                "vector": c_emb,
                "chunk_text": chunk_splits[i]
            } 
            for i, c_emb in enumerate(chunk_embs)
        ]
        
        # Create a new table for this story and configuration, and store the data
        # print(f"Creating table with name: {adjusted_story_id}-chunks at {story_db_dir}")
        db.create_table(
            f"{adjusted_story_id}-chunks",
            data=chunks_embs_data,
            mode="overwrite"  # Each story and configuration combination gets its own table
        )
        print("\n Creating new table for the story.")
    
def retrieve_best_matching_chunk(question, story_id, db_dir, top_n, embedding_model, similarity_func):
    db = lancedb.connect(db_dir)
    # print(f"Retrieve table: {story_id}-chunks")
    table = db.open_table(f"{story_id}-chunks")

    # Embed the question for the search
    question_embedding = embedding_model.embed_query(question)#[0].tolist()

    # Perform the search
    search_results = table.search(question_embedding) \
                            .metric(similarity_func) \
                            .select(["chunk_text", "vector"]) \
                            .limit(top_n) \
                            .to_pandas()
    
    # Check if the DataFrame is not empty
    if not search_results.empty:
        best_chunk_text = search_results.iloc[0]['chunk_text']  # Accessing the first row's chunk_text
        best_chunk_vector = search_results.iloc[0]['vector']  # Assuming 'vector' is stored and accessible
        return best_chunk_text, best_chunk_vector
    else:
        raise Exception("Empty chunk retrieved. No matching chunk found.")

# Function to normalize and stem text
def normalize_and_stem(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())  # Normalize to lowercase and tokenize
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]  # Stemming and removing punctuation
    return ' '.join(stemmed_tokens)

# Modified function to calculate the token-wise F1 score and return precision and recall
def token_eval(predicted, actual):
    predicted_tokens = predicted.split()
    actual_tokens = actual.split()
    common_tokens = Counter(predicted_tokens) & Counter(actual_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0 and len(predicted_tokens) == 0 and len(actual_tokens) == 0:
        # Case where both predicted and actual answers are empty
        return 1.0, 1.0, 1.0, 1  # Perfect score
    elif num_same == 0:
        return 0, 0, 0, 0  # Return zero precision, recall, F1 score, and exact match

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(actual_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    em = int(predicted.strip() == actual.strip())  # Exact match score

    return f1, precision, recall, em

def newsqa_loop(data, llm, output_csv_path, output_log_path, max_stories, chunk_sizes, overlap_percentages,
                instruct_embedding_model_name, instruct_embedding_model_kwargs, 
                instruct_embedding_encode_kwargs, top_n_chunk, db_dir, embedding_func, similarity_func):
    
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Chunk_size', 'Chunk_Overlap', 'Time', 'Story Number', 'Question Number', 'EM', 'Precision', 'Recall', 'F1', 'Error'])

        if embedding_func == 'hf_emb':
            # Embedding for story sentences
            story_embs = HuggingFaceInstructEmbeddings(
                model_name=instruct_embedding_model_name,
                model_kwargs=instruct_embedding_model_kwargs,
                encode_kwargs=instruct_embedding_encode_kwargs,
                embed_instruction="Use the following pieces of context to answer the question at the end:"
            )

            # Embedding for questions
            query_embs = HuggingFaceInstructEmbeddings(
                model_name=instruct_embedding_model_name,
                model_kwargs=instruct_embedding_model_kwargs,
                encode_kwargs=instruct_embedding_encode_kwargs,
                query_instruction="How does this information relate to the question?"
            )
        else:
            print("Unsupported embedding functions!")
        
        start_time = time.time()

        for chunk_size in chunk_sizes:
            print(f"\n{time.time()-start_time} Processing chunk size {chunk_size}:")
            last_time = time.time()
            
            for overlap_percentage in overlap_percentages:
                actual_overlap = int(chunk_size * overlap_percentage)
                print(f"\n{time.time()-start_time}\t{time.time()-last_time}\tOverlap [{overlap_percentage}] {actual_overlap}")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=actual_overlap)

                for i, story in enumerate(data['data']):
                    if i >= max_stories:
                        break

                    now_time = time.time()
                    print(f"\n{now_time - start_time}\t{now_time - last_time}\t\tstory {i + 1}: ", end='')
                    last_time = now_time
                    
                    # Define a unique directory name for the story with the specified chunk size and overlap
                    story_id = story['storyId']
                    adjusted_story_id = story_id[14:-6] if len(story_id) > 6 else story_id
                    unique_dir_name = f"{embedding_func}/chunk_{chunk_size}/overlap_{int(chunk_size * overlap_percentage)}/{adjusted_story_id}"
                    story_db_dir = f"{db_dir}/{unique_dir_name}"
                    print("\n In Loop:", story_db_dir)
                    process_and_store_chunks(
                        data=story, 
                        db_dir=story_db_dir,
                        chunk_size=chunk_size,
                        overlap_percentage=overlap_percentage,
                        text_splitter=text_splitter,
                        embedding_model=story_embs, 
                        embedding_func=embedding_func
                    )
                    
                    for j, question_data in enumerate(story['questions']):
                        if question_data['isAnswerAbsent']:
                            continue  # Skip this question because an answer is absent

                        question = question_data['q']

                        best_chunk_text, _ = retrieve_best_matching_chunk(question=question, 
                                                                                       story_id=adjusted_story_id, 
                                                                                       db_dir=story_db_dir, 
                                                                                       top_n=top_n_chunk,
                                                                                       embedding_model=query_embs, 
                                                                                       similarity_func=similarity_func)
                        # print(best_chunk_text)

                        best_chunk_emb = Chroma.from_texts(best_chunk_text, query_embs)

                        # Initialize the QA chain with the vectorstore as the retriever
                        qa_chain = RetrievalQA.from_chain_type(
                            llm, 
                            retriever=best_chunk_emb.as_retriever(), 
                            chain_type="stuff",
                            verbose=False,
                            # chain_type_kwargs={
                            #     "prompt": QA_CHAIN_PROMPT,
                            #     "verbose": False},
                            return_source_documents=False
                        )
                            
                        # Check if there is a consensus answer and extract it
                        consensus = question_data['consensus']
                        if 's' in consensus and 'e' in consensus:
                            actual_answer = story['text'][consensus['s']:consensus['e']]
                        else:
                            continue  # No consensus answer, skip to the next question

                        # Get the prediction from the model
                        # query = question # + "Please provide the answer in as few words as possible and do NOT repeat any word in the question."
                        query = f"""
                        Based on the following information only: 
                        
                        {best_chunk_text}
                        
                        {question} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "{question}".

                        Answer:
                        """
                        result = qa_chain.run(query)
                        print('q:', query, 'a:', result)
                        
                        # Extract and process the predicted answer
                        if result is None:
                            result = 0, 0, 0, 0
                        else:
                            print(f"Result from qa_chain.run(): {result}, type: {type(result)}")
                            predicted_answer = result if isinstance(result, str) else result['result']
                        
                        # Normalize and stem the predicted and actual answers
                        normalized_predicted_answer = normalize_and_stem(predicted_answer)
                        normalized_actual_answer = normalize_and_stem(actual_answer)

                        # Calculate the F1 score, precision, and recall using normalized and stemmed answers
                        # f1_score_value, precision, recall, em_score = token_eval(normalized_predicted_answer, normalized_actual_answer)
                        # print("Calling token_eval with:", normalized_predicted_answer, normalized_actual_answer)
                        result = token_eval(normalized_predicted_answer, normalized_actual_answer)
                        # print("token_eval returned:", result)
                        f1_score_value, precision, recall, em_score = result
                        
                        # Write the scores to the file
                        error = 1 if 'error' in normalized_predicted_answer else 0
                        if error==0: 
                            writer.writerow([chunk_size, overlap_percentage, embedding_func, similarity_func, top_n_chunk, time.time() - start_time, i, j, em_score, precision, recall, f1_score_value, error])
                        
                        with open(output_log_path, 'a') as details_file:
                            details_file.write(f"Chunk Size: {chunk_size}\n")
                            details_file.write(f"Overlap: {overlap_percentage}\n")
                            details_file.write(f"Embedding Func: {embedding_func}\n")
                            details_file.write(f"Similarity Func: {similarity_func}\n")
                            details_file.write(f"Top n Chunk Selected: {top_n_chunk}\n")
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
                    #del chunk_embs

                # End of the story loop
                # del text_splitter

############## Running Parameters ##############
max_stories = 50
random_seed = 123
db_dir = 'C:/NewsQA/lancedb'
embedding_func = 'hf_emb'
similarity_func = "cosine" # "cosine", "l2", "dot"
chunk_sizes = [200, 400]
# chunk_sizes = [50,25]
# overlap_percentages = [0, 0.1, 0.2]  # Expressed as percentages (0.1 = 10%)
overlap_percentages = [0, 0.1]
top_n_chunk = 1

# model_location = "C:/Users/24075/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"
model_location = "C:/NewsQA/GPT4ALL/mistral-7b-instruct-v0.1.Q4_0.gguf"
# model_location = "/Users/wk77/Library/CloudStorage/OneDrive-DrexelUniversity/Documents/data/gpt4all/models/gpt4all-falcon-q4_0.gguf"
# model_location = "/Users/wk77/Documents/data/gpt4all-falcon-newbpe-q4_0.gguf"
# model_location = "/Users/wk77/Documents/data/mistral-7b-instruct-v0.1.Q4_0.gguf"
input_file_path='C:/NewsQA/combined-newsqa-data-story2.json'
# input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/newsqa-data-v1.csv"
# input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined-newsqa-data-v1.json"
# input_file_path = "/Users/wk77/Documents/git/DeepDelight/Thread2/data/combined-newsqa-data-story1.json"
output_csv_path = '../results/combined_chunks3.csv'
# output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/story1_scores_test.csv"
# output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined_scores_test.csv"
output_log_path = '../results/combined_chunks3.log'

# # Initialize PairwiseDistance
# pdist = nn.PairwiseDistance(p=2.0, eps=1e-06)

##################################################
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)  # This will show only warnings and errors
logging.basicConfig(level=logging.ERROR)

print("Loading data.")
data = json.loads(Path(input_file_path).read_text())

# print("Setting template.")
# template_original = """
#                     Based on the following information only: 
                    
#                     {context}
                    
#                     {question} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. "{question}".

#                     Answer:
#                     """
# QA_CHAIN_PROMPT_ORIGINAL = PromptTemplate.from_template(template_original)

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
newsqa_loop(data, llm, output_csv_path, output_log_path, max_stories, chunk_sizes, overlap_percentages, 
            instruct_embedding_model_name, instruct_embedding_model_kwargs, instruct_embedding_encode_kwargs, 
            top_n_chunk, db_dir, embedding_func, similarity_func)
