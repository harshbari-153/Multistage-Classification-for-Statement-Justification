# install this library first before using it
# pip install vaderSentiment


###########################################################
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_polarity(sentence):
    # check for string data
    if not isinstance(sentence, str):
        return 0
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Analyze the sentiment of the sentence
    sentiment_scores = analyzer.polarity_scores(sentence)
    
    # Return the compound sentiment polarity score
    return sentiment_scores['compound']

# Example usage
# sentence = "I love this beautiful sunny day!"
# polarity_score = get_sentiment_polarity(sentence)
# print(f"Sentiment Polarity Score: {polarity_score}")

###########################################################


###########################################################

import re

# Function to load the MPQA Subjectivity Lexicon
def load_mpqa_lexicon(file_path):
    lexicon_mqpa = {}
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.split()
            if len(fields) >= 6:
                try:
                    word = fields[2].split('=')[1]
                    polarity = fields[5].split('=')[1]
                    lexicon_mqpa[word] = polarity
                except IndexError as e:
                    pass
            else:
                pass
            
    return lexicon_mqpa
    
# Load the MPQA lexicon
lexicon_mqpa = load_mpqa_lexicon('subjclueslen1-HLTEMNLP05.tff')

# Function to calculate sentiment polarity using MPQA Subjectivity Lexicon
def get_mpqa_sentiment_polarity(sentence):
    # check for string data
    if not isinstance(sentence, str):
        return 0
        
    words = re.findall(r'\w+', sentence.lower())
    score = 0
    for word in words:
        if word in lexicon_mqpa:
            if lexicon_mqpa[word] == 'positive':
                score += 1
            elif lexicon_mqpa[word] == 'negative':
                score -= 1
    return score

# Example usage
# sentence = "I love this beautiful sunny day!"
# polarity_score = get_mpqa_sentiment_polarity(sentence, lexicon)
# print(f"Sentiment Polarity Score: {polarity_score}")


###########################################################


###########################################################

import pandas as pd
import numpy as np
import re

# Function to load DepecheMood++ lexicon
def load_depechemood_lexicon(file_path):
    # Read the DepecheMood lexicon file into a DataFrame
    try:
        # Read the DepecheMood lexicon file into a DataFrame
        # df = pd.read_csv(file_path, sep='\t', header = 0, index_col = 'lemma')
        df = pd.read_csv(file_path, sept = '\t')
        # df.set_index('lemma', inplace=True)
        return df
    except Exception as e:
        return None
    
# Load the DepecheMood++ lexicon
# lexicon_emo = load_depechemood_lexicon('DepecheMood_english_token_full.tsv')

lexicon_emo = pd.read_csv('DepecheMood_english_token_full.tsv', sep = '\t', header = 0, index_col = 'lemma')

# Function to calculate sentiment polarity using DepecheMood++ lexicon
def get_depechemood_sentiment_polarity(sentence):
    # check for string data
    if not isinstance(sentence, str):
        return 0
    
    if lexicon_emo is None:
        print("Error: Lexicon is not loaded")
        quit()
        return None
        
    words = re.findall(r'\w+', sentence.lower())
    scores = []
    for word in words:
        if word in lexicon_emo.index:
            scores.append(lexicon_emo.loc[word].values)
    if scores:
        # Average the emotion scores for all words in the sentence
        mean_scores = np.mean(scores, axis=0)
        # Return the compound sentiment score as the sum of positive emotions minus the sum of negative emotions
        return mean_scores[2] + mean_scores[5] + mean_scores[6] + mean_scores[7] - mean_scores[8] - mean_scores[1] - mean_scores[3] - mean_scores[4]
    else:
        # If no words from the sentence are in the lexicon, return 0
        return 0

# Example usage
# sentence = "I love this beautiful sunny day!"
# polarity_score = get_depechemood_sentiment_polarity(sentence, lexicon)
# print(f"Sentiment Polarity Score: {polarity_score}")


###########################################################



file_name = "..\liar plus dataset\dataset\pre_processed_train.csv"
data = pd.read_csv(file_name)

# VADER sentiment polarity
data['s_pol'] = data['statement'].apply(lambda x: get_sentiment_polarity(x))
data['j_pol'] = data['justification'].apply(lambda x: get_sentiment_polarity(x))

print("VADER Sentiment Polarity Done")

# MPQA subjectivity lxicon
data['s_mpqa'] = data['statement'].apply(lambda x: get_mpqa_sentiment_polarity(x))
data['j_mpqa'] = data['justification'].apply(lambda x: get_mpqa_sentiment_polarity(x))

print("MQPA Subjectivity Polarity Done")

# Depechmood emotion EMO detection
data['s_emo'] = data['statement'].apply(lambda x: get_depechemood_sentiment_polarity(x))
print("Statement EMO done")
data['j_emo'] = data['justification'].apply(lambda x: get_depechemood_sentiment_polarity(x))

print("DepecheMood Emotion EMO Done")


# Save the updated DataFrame back to the CSV file
data.to_csv(file_name, index=False)

print("Done")