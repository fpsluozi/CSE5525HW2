# CSE5525 NLP Homework 2 Group 1
import nltk
import numpy
import itertools
from nltk.corpus import treebank

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# Step 2: Retrieve P(W_i | T_i) and P(T_i| T_i-1)
#
# Sample Usage 1: 
#	print set1_cpd_tags['DT'].prob('JJ')
# 	meaning print the prob of adjective given determinor from training set 1
#
# Sample Usage 2: 
#	print full_cpd_word_tag['DT'].prob('the')
# 	meaning print the prob of word 'the' given determinor from full training set
#
# PS. cpd as the Conditional Prob Distribution
# PSS. We use Laplace distribution for unseen cases

# Full Traning Set
full_training_set_words = []
for sent in full_training_set:
	full_training_set_words.append(('S','S'))
	full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
	full_training_set_words.append(('T','T'))

full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
full_cpd_word_tag = nltk.ConditionalProbDist(full_cfd_word_tag, nltk.SimpleGoodTuringProbDist)

full_tags = [tag for (tag, word) in full_training_set_words]
full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))
full_cpd_tags = nltk.ConditionalProbDist(full_cfd_tags, nltk.SimpleGoodTuringProbDist)

# Traning Set 1
set1_training_set_words = []
for sent in training_set1:
	set1_training_set_words.append(('S','S'))
	set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
	set1_training_set_words.append(('T','T'))

set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)
set1_cpd_word_tag = nltk.ConditionalProbDist(set1_cfd_word_tag, nltk.SimpleGoodTuringProbDist)

set1_tags = [tag for (tag, word) in set1_training_set_words]
set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))
set1_cpd_tags = nltk.ConditionalProbDist(set1_cfd_tags, nltk.SimpleGoodTuringProbDist)

print set1_cpd_tags['NN'].prob("T")

# Step 3

# Step 4

# Step 5

# Step 6

dict_tags = {}
dict_words = {}
num_tags = 1 # start from 1, consistent with equations in the book
num_words = 1
for sent in nltk.corpus.treebank.tagged_sents():
    for word_tag in sent:
        word = word_tag[0]
        tag = word_tag[1]
        if dict_words.get(word) == None:
            dict_words[word] = num_words
            num_words = num_words + 1
        if dict_tags.get(tag) == None:
            dict_tags[tag] = num_tags
            num_tags = num_tags + 1

#initalize from training_set1
A_table = numpy.zeros((num_tags, num_tags))
B_table = numpy.zeros((num_tags, num_words))

# aij
for tag_1 in dict_tags.keys():
    for tag_2 in dict_tags.keys():
        A_table[dict_tags[tag_1]][dict_tags[tag_2]] = set1_cpd_tags[tag_1].prob(tag_2)

# bi(W)
for tag in dict_tags.keys():
    for word in dict_words.keys():
        B_table[dict_tags[tag]][dict_words[word]] = set1_cpd_word_tag[tag].prob(word)

# a01 - a0T 
START_table = numpy.zeros(num_tags)
for tag in dict_tags.keys():
    START_table[dict_tags[tag]] = set1_cpd_tags['S'].prob(tag)

# a1(T+1) - aT(T+1)
END_table = numpy.zeros(num_tags)
for tag in dict_tags.keys():
    END_table[dict_tags[tag]] = set1_cpd_tags['T'].prob(tag)

# b(T+1)(END) = 1
B_end = 1.0

# running forward-backward training
# for every sentence in full_training_set

for sent in full_training_set:
    
    # get alpha
    T = len(sent)
    alpha_table = numpy.zeros((T + 1, num_tags)) # from 1, so add 1 here for convenience
    
    # intialize alpha_1j = a_0j * bj(W1)
    for i in xrange(1, num_tags):
        i = dict_tags[tag]
        j = dict_words[sent[0][0]] # Word1
        alpha_table[1][i] = START_table[i] * B_table[i][j]
    
    # compute alpha_tj
    for t in xrange(2, T+1):
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                Ot = dict_words[sent[t-1][0]]
                alpha_table[t][i] = alpha_table[t][i] + alpha_table[t-1][j] * A_table[j][i] * B_table[i][Ot]
    
    # compute P(O|lamda)
    P_O = 0
    for i in xrange(1, num_tags):
        P_O = P_O + alpha_table[T][i] * END_table[i]
        
    # get beta
    beta_table = numpy.zeros((T + 1, num_tags))
    
    #initialize beta_Ti
    for i in xrange(1, num_tags):
        beta_table[T][i] = END_table[i]
    
    # compute beta_ti
    for t in xrange(T-1, -1, -1):
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                Ot1 = dict_words[sent[t][0]]
                beta_table[t][i] = beta_table[t][i] + A_table[i][j] * B_table[j][Ot1] * beta_table[t+1][j]
    
    # compute sum_1-(T-1) (Xi_ij), no need to take alpha_TN into account, because it will be cancelled.
    XI_table = numpy.zeros((num_tags, num_tags))
    for t in xrange(1, T):
        Ot1 = dict_words[sent[t][0]]
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                XI_table[i][j] = XI_table[i][j] + alpha_table[t][i] * A_table[i][j] * B_table[t][Ot1] * beta_table[t+1][j]
    
    # compute sum_1-(T-1) (sum_1-N (Xi_ij))
    for i in xrange(1, num_tags):
        for j in xrange(1, num_tags):
            XI_table[i][0] = XI_table[i][0] + XI_table[i][j]
    
    # compute gamma_tj, no need to think about alpha_TN
    GAMMA_table = numpy.zeros((T+1, num_tags))
    for t in xrange(1, T+1):
        for j in xrange(1, num_tags):
            GAMMA_table[t][j] = alpha_table[t][j] * beta_table[t][j]
    
    # compute sum_1T(gamma_tj)
    for j in xrange(1, num_tags):
        for t in xrange(1, T+1):
            GAMMA_table[0][j] = GAMMA_table[0][j] + GAMMA_table[t][j]
    
    # compute new Aij:
    for i in xrange(1, num_tags):
        for j in xrange(1, num_tags):
            A_table[i][j] = XI_table[i][j] / XI_table[i][0]
    
    # compute new Bj(Vk): how to deal with other words not in the sentence? 0???
    for i in xrange(1, num_tags):
        for w in xrange(1, num_words):
            B_table[i][w] = 0.0   # is this right????
    
    for t in xrange(1, T+1):
        Ot = dict_words[sent[t][0]]
        for j in xrange(1, num_tags):
            B_table[j][Ot] = B_table[j][Ot] + GAMMA_table[t][j] / GAMMA_table[0][j]
            
    # test on the test set for every iteration??? (for every sentence?)
    # invoke viterbi algorithm  


