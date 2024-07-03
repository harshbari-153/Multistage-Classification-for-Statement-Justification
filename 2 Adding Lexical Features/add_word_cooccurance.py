import pandas as pd

def common_words_count(sentence1, sentence2):
    # Split sentences into words and convert to lowercase
    
    if type(sentence1) is not str or type(sentence2) is not str:
        return 0
        
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    
    # Find the intersection of the two sets
    common_words = words1.intersection(words2)
    
    # Count the number of common words
    return len(common_words)

# Load the dataset
file_name = '..\liar plus dataset\dataset\pre_processed_train.csv'
data = pd.read_csv(file_name)

i = 0
count = []
n = len(data['statement'])
for i in range(n):
  count.append(common_words_count(data['statement'].loc[i], data['justification'].loc[i]))
  i += 1

data['word_occurance'] = count

# Save the updated DataFrame back to the CSV file
data.to_csv(file_name, index=False)

print("Done")