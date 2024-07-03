import numpy as np
import pandas as pd

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

def preprocess_sentence(sentence):
    # Convert to lowercase and tokenize
    if type(sentence) is not str:
        return []
        
    words = sentence.lower().split()
    return words

def get_sentence_embedding(sentence, embeddings_index, embedding_dim=100):
    words = preprocess_sentence(sentence)
    valid_embeddings = []

    for word in words:
        if word in embeddings_index:
            valid_embeddings.append(embeddings_index[word])

    if not valid_embeddings:
        return np.zeros(embedding_dim)

    # Compute the average of the embeddings
    sentence_embedding = np.mean(valid_embeddings, axis=0)
    return sentence_embedding

# Path to the GloVe 100d file
glove_file_path = 'GloVe Embeddings\glove.6B.100d.txt'

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings(glove_file_path)

file_name = "..\liar plus dataset\dataset\pre_processed_train.csv"
data = pd.read_csv(file_name)

# Depechmood emotion EMO detection
data['s_gru'] = data['statement'].apply(lambda x: get_sentence_embedding(x, glove_embeddings))
data['j_gru'] = data['justification'].apply(lambda x: get_sentence_embedding(x, glove_embeddings))


# Save the updated DataFrame back to the CSV file
data.to_csv(file_name, index=False)

print("Done")

#################################################
# Input sentence
# sentence = "This is a sample sentence to generate GloVe embeddings."

# Get sentence embedding
# sentence_embedding = get_sentence_embedding(sentence, glove_embeddings)

# print("Sentence Embedding (100 dimensions):")
# print(sentence_embedding)
