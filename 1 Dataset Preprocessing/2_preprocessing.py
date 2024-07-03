import string
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run this only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text, forbidden_words, stop_words, stemmer):
  if isinstance(text, str):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stop words, and forbidden words
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in forbidden_words]
    
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

# Read the input CSV file
input_path = '..\liar plus dataset\dataset\\'
input_file = 'val.csv'
data = pd.read_csv(input_file)

# Read the forbidden words from the text file
with open('..\liar plus dataset\forbidden_words.txt', 'r') as file:
    forbidden_words = set(word.strip() for word in file.readlines())

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Apply preprocessing to both attributes 'A' and 'B'
data['statement'] = data['statement'].apply(lambda x: preprocess_text(x, forbidden_words, stop_words, stemmer))
data['justification'] = data['justification'].apply(lambda x: preprocess_text(x, forbidden_words, stop_words, stemmer))

# Save the preprocessed data to a new CSV file
data_output = data[['statement', 'justification', 'label']]

output_file = input_path + 'pre_processed_' + input_file
data_output.to_csv(output_file, index=False)


print(f"{output_file} preprocessing done")