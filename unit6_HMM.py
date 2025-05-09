#ENCANTO AND SONCIO EXERCISE 6 NO.1 OUTPUT

#import libraries
from collections import defaultdict, Counter

#training set definition
dataset = [
    [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    [("The", "DET"), ("dog", "NOUN"), ("sleeps", "VERB")],
    [("My", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB"), ("loudly", "ADV")],
    [("Your", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("The", "DET"), ("bird", "NOUN"), ("sings", "VERB"), ("sweetly", "ADV")],
    [("A", "DET"), ("bird", "NOUN"), ("chirps", "VERB")]
]

#initialization of data structures
states = set() #for POS tags
vocab = set() #for word vocabs
init_counts = Counter() #initial tags
trans_counts = defaultdict(Counter) #transition tags
emit_counts = defaultdict(Counter) #emission tags
state_counts = Counter() #total counts of each tag

#processing training set
for sentence in dataset: #loop through each sentences in the training set
    previous_tag = None
    for i, (word, tag) in enumerate(sentence): #loop through each word tag pair
        vocab.add(word.lower()) #add words to the vocab
        states.add(tag)  #adds tag to the state
        emit_counts[tag][word.lower()] += 1
        state_counts[tag] += 1
        if i == 0:
            init_counts[tag] += 1
        if previous_tag is not None:
            trans_counts[previous_tag][tag] += 1
        previous_tag = tag

def normalize(counter): #converting counts to probabilities
    total = sum(counter.values())
    return {key: val / total for key, val in counter.items()}

init_probs = normalize(init_counts) #probability of each tag starting a sentence
trans_probs = {tag: normalize(trans_counts[tag]) for tag in trans_counts} #transition probability
emit_probs = {tag: normalize(emit_counts[tag]) for tag in emit_counts} #emission probability
