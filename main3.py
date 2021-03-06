# CSE5525 NLP Homework 2 Group 1
# This file includes Step 1, step 2 and a simple Viterbi test out of Step 3
import nltk
import numpy
import math
import itertools
from nltk.corpus import treebank
from viterbi import viterbi

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# Step 2: Retrieve P(W_i | T_i) and P(T_i| T_i-1)
# Full Traning Set
full_training_set_words = []
for sent in full_training_set:
    full_training_set_words.append(('<s>','<s>'))
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
    full_training_set_words.append(('</s>','</s>'))

full_tags = [tag for (tag, word) in full_training_set_words]
full_words = [word for (tag, word) in full_training_set_words]

full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))
    
# Get all the tags and words


full_tags = [tag for (tag, word) in full_training_set_words]
full_words = [word for (tag, word) in full_training_set_words]
    
full_tag_set = set(full_tags)
full_word_set = set(full_words)

num_tags = len(full_tag_set)
num_words = len(full_word_set)

dict_tags = {}
dict_words = {}

dict_tags['<s>'] = 0
dict_words['<s>'] = 0
temp = 1
for tag in full_tag_set:
    if tag != '<s>' and tag != '</s>':
        dict_tags[tag] = temp
        temp = temp + 1
temp = 1
for word in full_word_set:
    if word != '<s>' and word != '</s>':
        dict_words[word] = temp
        temp = temp + 1 
dict_tags['</s>'] = num_tags - 1
dict_words['</s>'] = num_words - 1

# Traning Set 1
set1_training_set_words = []
for sent in training_set1:
    set1_training_set_words.append(('<s>','<s>'))
    set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
    set1_training_set_words.append(('</s>','</s>'))
    
set1_tags = [tag for (tag, word) in set1_training_set_words]
set1_words = [word for (tag, word) in set1_training_set_words]

set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)
set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))

# eliminate wrong counts
full_cfd_tags['</s>']['<s>'] = 0
set1_cfd_tags['</s>']['<s>'] = 0

# Get the transition prob
A_full_table = numpy.zeros((num_tags, num_tags), dtype = 'double')
A_set1_table = numpy.zeros((num_tags, num_tags), dtype = 'double')

for tag_1 in full_tag_set:
    if tag_1 != '</s>':
        full_num = 0
        set1_num = 0
        for tag_2 in full_tag_set:
            if tag_2 != '<s>' and (tag_1 != '<s>' or tag_2 != '</s>'):
                if full_cfd_tags[tag_1][tag_2] == 0:
                    full_cfd_tags[tag_1][tag_2] = 1
                if set1_cfd_tags[tag_1][tag_2] == 0:
                    set1_cfd_tags[tag_1][tag_2] = 1
                full_num = full_num + full_cfd_tags[tag_1][tag_2]
                set1_num = set1_num + set1_cfd_tags[tag_1][tag_2]
        for tag_2 in full_tag_set:
            A_full_table[dict_tags[tag_1]][dict_tags[tag_2]] = float(full_cfd_tags[tag_1][tag_2])/float(full_num)
            A_set1_table[dict_tags[tag_1]][dict_tags[tag_2]] = float(set1_cfd_tags[tag_1][tag_2])/float(set1_num)
            
# Get the emission probs
B_full_table = numpy.zeros((num_tags, num_words), dtype = 'double')
B_set1_table = numpy.zeros((num_tags, num_words), dtype = 'double')

B_full_table[dict_tags['<s>']][dict_words['<s>']] = 1.0
B_full_table[dict_tags['</s>']][dict_words['</s>']] = 1.0
B_set1_table[dict_tags['<s>']][dict_words['<s>']] = 1.0
B_set1_table[dict_tags['</s>']][dict_words['</s>']] = 1.0

for tag in full_tag_set:
    if tag != '</s>' and tag != '<s>':
        full_num = 0
        set1_num = 0
        for word in full_word_set:
            if word != '</s>' and word != '<s>':
                if full_cfd_word_tag[tag][word] == 0:
                    full_cfd_word_tag[tag][word] = 1
                if set1_cfd_word_tag[tag][word] == 0:
                    set1_cfd_word_tag[tag][word] = 1
                full_num = full_num + full_cfd_word_tag[tag][word]
                set1_num = set1_num + set1_cfd_word_tag[tag][word]
        for word in full_word_set:
            B_full_table[dict_tags[tag]][dict_words[word]] = float(full_cfd_word_tag[tag][word])/float(full_num)
            B_set1_table[dict_tags[tag]][dict_words[word]] = float(set1_cfd_word_tag[tag][word])/float(set1_num)


# Step 3 Viterbi

# The init_table is the initial probabilities and enforces every sentence to start with <s>
init_table = {}
for tag in full_tag_set:
    init_table[tag] = 0.00000000000000000001
init_table['<s>'] = 1.0

# test_obs is a sentence to test viterbi
test_obs = ['<s>', 'Pierre', 'Vinken', ',' , '61' , "years", "old", "will", "join", "the", "board", "as", "a", "nonexecutive", "director", "Nov.", "29","."]

#run viterbi
print viterbi(dict_tags, dict_words, test_obs, full_tag_set, init_table, A_full_table, B_full_table )


    


