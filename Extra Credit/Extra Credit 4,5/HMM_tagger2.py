import nltk
from nltk.corpus import conll2000
import numpy
import math
import itertools
from nltk.corpus import treebank

# get the chunked sents from conll2000 corpus
# generate the train_data and test_data

training_set = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_set = conll2000.chunked_sents('test.txt', chunk_types=['NP'])


train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in training_set]

test_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in test_set]

#
# Compute the probs, similar as step2
#

full_training_set = []
for sent in train_data:
    full_training_set.append(('<s>','<s>'))
    full_training_set.extend([ (chunk, tag) for (tag, chunk) in sent])
    full_training_set.append(('</s>','</s>'))

full_tags = [tag for (chunk, tag) in full_training_set]
full_chunks = [chunk for (chunk, tag) in full_training_set]

full_cfd_tag_chunk = nltk.ConditionalFreqDist(full_training_set)
full_cfd_chunk = nltk.ConditionalFreqDist(nltk.bigrams(full_chunks))
    
# Get all the tags and chunks
for sent in test_data:
    full_training_set.append(('<s>','<s>'))
    full_training_set.extend([ (chunk, tag) for (tag, chunk) in sent ])
    full_training_set.append(('</s>','</s>'))

full_tags = [tag for (chunk, tag) in full_training_set]
full_chunks = [chunk for (chunk, tag) in full_training_set]
    
full_tag_set = set(full_tags)
full_chunk_set = set(full_chunks)

num_tags = len(full_tag_set)
num_chunks = len(full_chunk_set)

dict_tags = {}
dict_chunks = {}

dict_tags['<s>'] = 0
dict_chunks['<s>'] = 0
temp = 1
for tag in full_tag_set:
    if tag != '<s>' and tag != '</s>':
        dict_tags[tag] = temp
        temp = temp + 1
temp = 1
for chunk in full_chunk_set:
    if chunk != '<s>' and chunk != '</s>':
        dict_chunks[chunk] = temp
        temp = temp + 1 
dict_tags['</s>'] = num_tags - 1
dict_chunks['</s>'] = num_chunks - 1

# eliminate wrong counts
full_cfd_chunk['</s>']['<s>'] = 0

# Get the transition prob
A_full_table = numpy.zeros((num_chunks, num_chunks), dtype = 'double')

for chunk_1 in full_chunk_set:
    if chunk_1 != '</s>':
        full_num = 0
        for chunk_2 in full_chunk_set:
            if chunk_2 != '<s>' and (chunk_1 != '<s>' or chunk_2 != '</s>'):
                if full_cfd_chunk[chunk_1][chunk_2] == 0:
                    full_cfd_chunk[chunk_1][chunk_2] = 1
                full_num = full_num + full_cfd_chunk[chunk_1][chunk_2]
        for chunk_2 in full_chunk_set:
            A_full_table[dict_chunks[chunk_1]][dict_chunks[chunk_2]] = float(full_cfd_chunk[chunk_1][chunk_2])/float(full_num)
            
# Get the emission probs
B_full_table = numpy.zeros((num_chunks, num_tags), dtype = 'double')

B_full_table[dict_chunks['<s>']][dict_tags['<s>']] = 1.0
B_full_table[dict_chunks['</s>']][dict_tags['</s>']] = 1.0

for chunk in full_chunk_set:
    if chunk != '</s>' and chunk != '<s>':
        full_num = 0
        for tag in full_tag_set:
            if tag != '</s>' and tag != '<s>':
                if full_cfd_tag_chunk[chunk][tag] == 0:
                    full_cfd_tag_chunk[chunk][tag] = 1
                full_num = full_num + full_cfd_tag_chunk[chunk][tag]
        for tag in full_tag_set:
            B_full_table[dict_chunks[chunk]][dict_tags[tag]] = float(full_cfd_tag_chunk[chunk][tag])/float(full_num)

# build the HMM_tagger

full_tag_set.remove('<s>')
full_tag_set.remove('</s>')
full_chunk_set.remove('<s>')
full_chunk_set.remove('</s>') 

import fst
import math

eps = '¦Å'

HMM_tagger = fst.Transducer()

num_temp = num_chunks - 2

for chunk in full_chunk_set:
    HMM_tagger.add_arc(0, dict_chunks[chunk], eps, eps, -math.log(A_full_table[0][dict_chunks[chunk]]))

for chunk in full_chunk_set:
    i = dict_chunks[chunk]
    for tag in full_tag_set:
        HMM_tagger.add_arc(i, num_temp + i, tag, chunk, -math.log(B_full_table[i][dict_tags[tag]]))

for chunk1 in full_chunk_set:
    i = dict_chunks[chunk1]
    for chunk2 in full_chunk_set:
        j = dict_chunks[chunk2]
        HMM_tagger.add_arc(num_temp + i, j, eps, eps, -math.log(A_full_table[i][j]))

for chunk in full_chunk_set:
    i = dict_chunks[chunk]
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

for (tag, chunk) in test_data[index]:
    test.add_arc(num_temp, num_temp + 1, tag)
    num_temp = num_temp + 1
    right_ans.append(chunk)
test[num_temp].final = True

test = ((test >> HMM_tagger).shortest_path())
test.project_output()

my_ans = []

for state in test.states:
    for arc in state.arcs:
        if arc.ilabel != 0:
            my_ans.insert(0, HMM_tagger.osyms.find(arc.ilabel))

print "test sentence: ", test_data[index]
print
print "right tags: ", right_ans
print
print "my    tags: ", my_ans