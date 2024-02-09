import requests

def count_words_from_url(url):
    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            text = response.text
            words = text.split()
            unique_words = set(words)
            return len(words), len(unique_words)
        else:
            return f"Failed to download the data. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
total_words, unique_words = count_words_from_url(data_url)
print(f"Total words: {total_words}")
print(f"Unique words: {unique_words}")
