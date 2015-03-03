# CSE5525 NLP Homework 2 Group 1
import nltk
import numpy
import itertools
from nltk.corpus import treebank

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# 
# Get the probability tables for the HMM tagger.
# Similar as step 2, but eliminate some redundant parts.
#

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
for sent in test_set:
    full_training_set_words.append(('<s>','<s>'))
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
    full_training_set_words.append(('</s>','</s>'))

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

# eliminate wrong counts
full_cfd_tags['</s>']['<s>'] = 0

# Get the transition prob
A_full_table = numpy.zeros((num_tags, num_tags), dtype = 'double')

for tag_1 in full_tag_set:
    if tag_1 != '</s>':
        full_num = 0
        for tag_2 in full_tag_set:
            if tag_2 != '<s>' and (tag_1 != '<s>' or tag_2 != '</s>'):
                if full_cfd_tags[tag_1][tag_2] == 0:
                    full_cfd_tags[tag_1][tag_2] = 1
                full_num = full_num + full_cfd_tags[tag_1][tag_2]
        for tag_2 in full_tag_set:
            A_full_table[dict_tags[tag_1]][dict_tags[tag_2]] = float(full_cfd_tags[tag_1][tag_2])/float(full_num)
            
# Get the emission probs
B_full_table = numpy.zeros((num_tags, num_words), dtype = 'double')

B_full_table[dict_tags['<s>']][dict_words['<s>']] = 1.0
B_full_table[dict_tags['</s>']][dict_words['</s>']] = 1.0

for tag in full_tag_set:
    if tag != '</s>' and tag != '<s>':
        full_num = 0
        for word in full_word_set:
            if word != '</s>' and word != '<s>':
                if full_cfd_word_tag[tag][word] == 0:
                    full_cfd_word_tag[tag][word] = 1
                full_num = full_num + full_cfd_word_tag[tag][word]
        for word in full_word_set:
            B_full_table[dict_tags[tag]][dict_words[word]] = float(full_cfd_word_tag[tag][word])/float(full_num)
            
full_tag_set.remove('<s>')
full_tag_set.remove('</s>')
full_word_set.remove('<s>')
full_word_set.remove('</s>') 

# build the HMM_tagger

import fst
import math

eps = '¦Å'

HMM_tagger = fst.Transducer()

num_temp = num_tags - 2

for tag in full_tag_set:
    HMM_tagger.add_arc(0, dict_tags[tag], eps, eps, -math.log(A_full_table[0][dict_tags[tag]]))

for tag in full_tag_set:
    i = dict_tags[tag]
    for word in full_word_set:
        HMM_tagger.add_arc(i, num_temp + i, word, tag, -math.log(B_full_table[i][dict_words[word]]))

for tag1 in full_tag_set:
    i = dict_tags[tag1]
    for tag2 in full_tag_set:
        j = dict_tags[tag2]
        HMM_tagger.add_arc(num_temp + i, j, eps, eps, -math.log(A_full_table[i][j]))

for tag in full_tag_set:
    i = dict_tags[tag]
    HMM_tagger.add_arc(num_temp + i, 2 * num_temp +1, eps, eps, -math.log(A_full_table[i][num_temp + 1]))

HMM_tagger[num_temp*2 + 1].final = True


###########################################################
# test part 
# you can run the above part first, and run the test part for many times, and change the index at each time.
# use sents in test_set and output the sentences, right ans and my ans.
###########################################################

test = fst.Acceptor(HMM_tagger.isyms)
num_temp = 0

# index of testing sentence
index = 0

right_ans = []

for (word, tag) in test_set[index]:
    test.add_arc(num_temp, num_temp + 1, word)
    num_temp = num_temp + 1
    right_ans.append(tag)
test[num_temp].final = True

test = ((test >> HMM_tagger).shortest_path())
test.project_output()

my_ans = []

for state in test.states:
    for arc in state.arcs:
        if arc.ilabel != 0:
            my_ans.insert(0, HMM_tagger.osyms.find(arc.ilabel))
            
print "test sentence: ", test_set[index]
print
print "right tags: ", right_ans
print
print "my    tags: ", my_ans