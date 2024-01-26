
import logging
import time
from collections import Counter
from collections import defaultdict
import csv
import json
from langchain_community.llms import GPT4All
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from transformers import set_seed
from langchain_community.embeddings import GPT4AllEmbeddings

# Helper function to calculate Exact Match (EM) score
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

def newsqa_loop(data, llm, output_file_path, chunk_sizes, overlap_percentages, max_stories, instruct_embedding_model_name, instruct_embedding_model_kwargs, instruct_embedding_encode_kwargs, QA_CHAIN_PROMPT):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Chunk_size', 'Chunk_Overlap', 'Time', 'Story Number', 'Question Number', 'EM', 'Precision', 'Recall', 'F1'])

        word_embed = HuggingFaceInstructEmbeddings(
            model_name=instruct_embedding_model_name,
            model_kwargs=instruct_embedding_model_kwargs,
            encode_kwargs=instruct_embedding_encode_kwargs
        )
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
                    print(f"\n{now_time-start_time}\t{now_time-last_time}\t\tstory {i+1}: ", end='')
                    last_time = now_time
                    
                    all_splits = text_splitter.split_text(story['text'])
                    vectorstore = Chroma.from_texts(texts=all_splits, embedding=word_embed)
                    qa_chain = RetrievalQA.from_chain_type(
                                llm, 
                                retriever=vectorstore.as_retriever(), 
                                chain_type="stuff",
                                verbose=False,
                                chain_type_kwargs={
                                    "prompt": QA_CHAIN_PROMPT,
                                    "verbose": False},
                                return_source_documents=False)
                    
                    chunk_boundaries = []
                    start_index = 0
                    for split in all_splits:
                        end_index = start_index + len(split)
                        chunk_boundaries.append((start_index, end_index))
                        start_index = end_index

                    for j, question_data in enumerate(story['questions']):
                        if question_data['isAnswerAbsent']:
                            continue  # Skip this question because an answer is absent
                        
                        question = question_data['q']
                        consensus = question_data['consensus']
                        
                        # Check if there is a consensus answer and extract it
                        if 's' in consensus and 'e' in consensus:
                            actual_answer = story['text'][consensus['s']:consensus['e']]
                            answer_chunk_index = next((index for index, (start, end) in enumerate(chunk_boundaries) if consensus['s'] >= start and consensus['e'] <= end), None)
                            if answer_chunk_index is not None:
                                context_for_qa = all_splits[answer_chunk_index]
                        else:
                            continue  # No consensus answer, skip to the next question

                        # Get the prediction from the model
                        result = qa_chain({"context": context_for_qa, "query": question})
                        
                        # Extract and process the predicted answer
                        predicted_answer = result['result'] if isinstance(result['result'], str) else ""

                        # Calculate the F1 score, precision, and recall
                        f1_score_value, precision, recall = calculate_token_f1(predicted_answer, actual_answer)
                        em_score = calculate_em(predicted_answer, actual_answer)

                        # Write the scores to the file
                        writer.writerow([chunk_size, overlap_percentage, time.time() - start_time, i, j, em_score, precision, recall, f1_score_value])

                    # Cleanup
                    del qa_chain
                    del vectorstore
                    del all_splits

                # End of the story loop
                del text_splitter

############## Running Parameters ##############
max_stories = 100
# chunk_sizes = [200, 400, 800, 1600]
# overlap_percentages = [0, 0.1, 0.2, 0.4]  # Expressed as percentages (0.1 = 10%)
chunk_sizes = [50, 100, 150, 200, 300, 400]
overlap_percentages = [0, 0.1, 0.2, 0.3, 0.4]  # Expressed as percentages (0.1 = 10%)
random_seed = 123
# model_location = "C:/Users/24075/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"
# model_location = "/Users/wk77/Library/CloudStorage/OneDrive-DrexelUniversity/Documents/data/gpt4all/models/gpt4all-falcon-q4_0.gguf"
# model_location = "/Users/wk77/Documents/data/gpt4all-falcon-newbpe-q4_0.gguf"
model_location = "/Users/wk77/Documents/data/mistral-7b-instruct-v0.1.Q4_0.gguf"
# input_file_path='C:/NewsQA/combined-newsqa-data-story2.json'
# input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/newsqa-data-v1.csv"
input_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined-newsqa-data-v1.json"
# input_file_path = "/Users/wk77/Documents/git/DeepDelight/Thread2/data/combined-newsqa-data-story1.json"
# output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/story1_scores_test.csv"
output_file_path = "/Users/wk77/Documents/data/newsqa-data-v1/combined_scores_test.csv"

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
# instruct_embedding_model_kwargs = {'device': 'cpu'}
instruct_embedding_model_kwargs = {'device': 'mps'}
instruct_embedding_encode_kwargs = {'normalize_embeddings': True}

# The following code would iterate over the stories and questions to calculate the scores
start_time = time.time()
print(f"{start_time} Started.")                

# Main Function Execution
print("Processing.")
newsqa_loop(data, llm, output_file_path, chunk_sizes, overlap_percentages, max_stories, instruct_embedding_model_name,
            instruct_embedding_model_kwargs, instruct_embedding_encode_kwargs, QA_CHAIN_PROMPT_ORIGINAL)
