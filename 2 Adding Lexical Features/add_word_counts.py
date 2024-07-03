import pandas as pd

rft = ["fake", "fraud", "hoax", "deny", "disprove", "debunk", "refute", "contradict", "contest", "challenge", "repudiate", "discredit", "invalidate", "counter", "rebut", "disclaim", "oppose", "reject", "disavow", "nullify"]

cue = ["uncertain", "unsure", "skeptical", "dubious", "questionable", "suspect", "ambiguous", "vague", "inconclusive", "tentative", "deny", "disavow", "reject", "refuse", "repudiate", "renounce", "contradict", "oppose", "disclaim", "withhold", "fake", "fraud", "hoax", "counterfeit", "sham", "phony", "bogus", "deceptive", "forged", "spurious", "no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "negative"]

hdg = ["about", "claim", "essentially", "perhaps", "possibly", "probably", "apparently", "arguably", "seemingly", "roughly", "approximately", "likely", "may", "might", "could", "suggest", "indicate", "appear", "assume", "estimate", "generally", "typically", "presumably", "supposedly", "more or less", "kind of", "sort of", "around", "often", "sometimes", "maybe", "somewhat"]

imp = ["manage", "misfortune", "decline", "imply", "suggest", "hint", "indicate", "infer", "insinuate", "assume", "suspect", "propose", "allege", "contend", "assert", "maintain", "argue", "claim", "presume", "purport", "intimate", "betoken", "connote", "presuppose", "entail", "denote", "signal", "signify", "mean"]

file_name = "D:\SVNIT\Dissertation\Multistage Classification\liar plus dataset\dataset\pre_processed_test.csv"
data = pd.read_csv(file_name)

# Function to count words in a list
def count_words_in_list(words, word_list):
    if pd.isna(words):
        return 0
    return sum(word in word_list for word in str(words).split())

# Statement Counts
data['s_rft'] = data['statement'].apply(lambda x: count_words_in_list(x, rft))
data['s_cue'] = data['statement'].apply(lambda x: count_words_in_list(x, cue))
data['s_hdg'] = data['statement'].apply(lambda x: count_words_in_list(x, hdg))
data['s_imp'] = data['statement'].apply(lambda x: count_words_in_list(x, imp))

# Justification Counts
data['j_rft'] = data['justification'].apply(lambda x: count_words_in_list(x, rft))
data['j_cue'] = data['justification'].apply(lambda x: count_words_in_list(x, cue))
data['j_hdg'] = data['justification'].apply(lambda x: count_words_in_list(x, hdg))
data['j_imp'] = data['justification'].apply(lambda x: count_words_in_list(x, imp))

# Save the updated DataFrame back to the CSV file
data.to_csv(file_name, index=False)

print("Done")