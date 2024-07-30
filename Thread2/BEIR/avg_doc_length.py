import json

# Function to read JSONL file and calculate average document length
def calculate_avg_doc_length(file_path):
    total_length = 0
    total_docs = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            text_length = len(doc['text'].split())
            total_length += text_length
            total_docs += 1
    
    avg_doc_length = total_length / total_docs if total_docs > 0 else 0
    return avg_doc_length

# Path to the JSONL file
file_path = './BEIR_example_results/datasets/scifact/corpus.jsonl'

# Calculate and print the average document length
avg_length = calculate_avg_doc_length(file_path)
print(f"Average Document Length: {avg_length:.2f} words")

# The average length for scifact dataset corpus is 201.81 words.
