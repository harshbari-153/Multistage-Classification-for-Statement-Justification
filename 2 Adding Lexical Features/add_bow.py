import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_name = '..\liar plus dataset\dataset\pre_processed_train.csv'
df = pd.read_csv(file_name)  # Update with your file path

# Fill NaN values with an empty string
df['statement'] = df['statement'].fillna('')
df['justification'] = df['justification'].fillna('')

# Combine attr1 and attr2 into a single list for vectorization
combined_text = df['statement'].tolist() + df['justification'].tolist()

# Initialize CountVectorizer with 1-gram, 2-gram, and 3-gram
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2000)

# Fit and transform the combined text data
X = vectorizer.fit_transform(combined_text)

# Split the transformed data back into attr1 and attr2
attr1_bow = X[:len(df)]
attr2_bow = X[len(df):]

# Calculate cosine similarity between attr1 and attr2
cosine_similarities = []
for i in range(len(df)):
    sim = cosine_similarity(attr1_bow[i], attr2_bow[i])[0][0]
    cosine_similarities.append(sim)

# Add the cosine similarity to a new column attr3
df['cosin_sim'] = cosine_similarities

# Save the updated DataFrame to a CSV file
df.to_csv(file_name, index=False)  # Update with your desired file path

print("Done")
