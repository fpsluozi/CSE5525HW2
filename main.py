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
#   print set1_cpd_tags['DT'].prob('JJ')
#   meaning print the prob of adjective given determinor from training set 1
#
# Sample Usage 2: 
#   print full_cpd_word_tag['DT'].prob('the')
#   meaning print the prob of word 'the' given determinor from full training set
#
# PS. cpd as the Conditional Prob Distribution
# PSS. We use Laplace distribution for unseen cases

# Full Traning Set
full_training_set_words = []
for sent in full_training_set:
    full_training_set_words.append(('<s>','<s>'))
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
    full_training_set_words.append(('</s>','</s>'))

full_tags = [tag for (tag, word) in full_training_set_words]
full_words = [word for (tag, word) in full_training_set_words]
full_tag_set = set(full_tags)

full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))

full_obs_set = []
for sent in full_training_set:
    full_obs_set.append([ word for (word, tag) in sent ])
for i in xrange(len(full_obs_set)):
    full_obs_set[i].append('</s>')
    full_obs_set[i].insert(0, '<s>')

"""
for tag_1 in set(full_tags): # Laplace smoothing
    for tag_2 in set(full_tags):
        if full_cfd_tags[tag_1][tag_2] == 0:
            full_cfd_tags[tag_1][tag_2] = 1
for tag in set(full_tags): # Laplace smoothing
    for word in set(full_words):
        if full_cfd_word_tag[tag][word] == 0:
            full_cfd_word_tag[tag][word] = 1
"""
            
full_cpd_word_tag = nltk.ConditionalProbDist(full_cfd_word_tag, nltk.MLEProbDist)
full_cpd_tags = nltk.ConditionalProbDist(full_cfd_tags, nltk.MLEProbDist)

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

"""
for tag_1 in set(set1_tags): # Laplace smoothing
    for tag_2 in set(set1_tags):
        if set1_cfd_tags[tag_1][tag_2] == 0:
            set1_cfd_tags[tag_1][tag_2] = 1
for tag in set(set1_tags): # Laplace smoothing
    for word in set(set1_words):
        if set1_cfd_word_tag[tag][word] == 0:
            set1_cfd_word_tag[tag][word] = 1
"""
            
set1_cpd_word_tag = nltk.ConditionalProbDist(set1_cfd_word_tag, nltk.MLEProbDist)
set1_cpd_tags = nltk.ConditionalProbDist(set1_cfd_tags, nltk.MLEProbDist)

# Step 3 Viterbi
# Since both emission prob and transition prob can be zero, we use an extremely small epsilon
# to rule out zero probs during log calculation.
from viterbi import viterbi

# The init_table is the initial probabilities and enforces every sentence to start with <s>
init_table = {}
for tag in full_tag_set:
    init_table[tag] = 0.00000000000000000001
init_table['<s>'] = 1.0

# test_obs is a sentence to test viterbi
test_obs = ['<s>', 'Pierre', 'Viken', ',' , '61' , "years", "old", "will", "join", "the", "board", "as", "a", "nonexecutive", "director", "Nov.", "29",".", "</s>"]
#run viterbi
print viterbi(test_obs, full_tag_set, init_table, full_cpd_tags, full_cpd_word_tag )

# Step 4

# Step 5

# Step 6

import numpy

epsilon = 0.001
dict_tags = {}
dict_words = {}
num_tags = 1 # start from 1, consistent with equations in the book
num_words = 1

for tag in set(full_tags):
    dict_tags[tag] = num_tags
    num_tags = num_tags + 1
    
for word in set(full_words):
    dict_words[word] = num_words
    num_words = num_words + 1
            
#initalize from training_set1
A_table = numpy.zeros((num_tags + 1, num_tags + 1), dtype='double')
B_table = numpy.zeros((num_tags + 1, num_words + 1), dtype='double')

# aij
for tag_1 in dict_tags.keys():
    for tag_2 in dict_tags.keys():
        A_table[dict_tags[tag_1]][dict_tags[tag_2]] = set1_cpd_tags[tag_1].prob(tag_2)

# bi(W)
for tag in dict_tags.keys():
    for word in dict_words.keys():
        B_table[dict_tags[tag]][dict_words[word]] = set1_cpd_word_tag[tag].prob(word)

# a01 - a0T 
#START_table = numpy.zeros(num_tags)
for tag in dict_tags.keys():
    A_table[0][dict_tags[tag]] = set1_cpd_tags['<s>'].prob(tag)

# a1(T+1) - aT(T+1)
#END_table = numpy.zeros(num_tags)
for tag in dict_tags.keys():
    A_table[dict_tags[tag]][num_tags] = set1_cpd_tags[tag].prob('</s>')

# b(T+1)(END) = b(0)(START) = 1
B_table[0][0] = 1
B_table[num_tags][num_words] = 1

converged = False

while not converged:
    
    converged = True
    
    # running forward-backward training
    # for every sentence in full_training_set as one iteration.

    # create Xi, Gamma table
    XI_table = numpy.zeros((num_tags, num_tags + 1), dtype='double')
    GAMMA_table = numpy.zeros((num_tags + 1, num_words + 1), dtype='double')
    
    #for sent in full_training_set:
    sent = full_training_set[0]
    # get alpha
    T = len(sent)
    alpha_table = numpy.zeros((T + 2, num_tags + 1), dtype='double') 

    # alpha_00 = 1
    alpha_table[0][0] = 1

    # intialize alpha_1j = a_0j * bj(W1)
    for i in xrange(1, num_tags):
        i = dict_tags[tag]
        j = dict_words[sent[0][0]] # Word1
        alpha_table[1][i] = A_table[0][i] * B_table[i][j]

    # compute alpha_tj
    for t in xrange(2, T+1):
        Ot = dict_words[sent[t-1][0]]
        print t
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                alpha_table[t][i] = alpha_table[t][i] + alpha_table[t-1][j] * A_table[j][i] * B_table[i][Ot]

    # compute alpha_(T+1)N
    for i in xrange(1, num_tags):
        alpha_table[T+1][num_tags] = alpha_table[T+1][num_tags] + alpha_table[T][i] * A_table[i][num_tags]

    # compute P(O|lamda) = aplha_(T+1)N
    #P_O = 0
    #for i in xrange(1, num_tags):
    #   P_O = P_O + alpha[T][i] * END_table[i]
    P_O = alpha_table[T+1][num_tags]
    print P_O

    # get beta
    beta_table = numpy.zeros((T + 2, num_tags + 1), dtype='double')

    # beta_(T+1)N = 1
    beta_table[T+1][num_tags] = 1

    #initialize beta_Ti = aiN
    for i in xrange(1, num_tags):
        beta_table[T][i] = A_table[i][num_tags]

    # compute beta_ti
    for t in xrange(T-1, 0, -1):
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                Ot1 = dict_words[sent[t][0]]
                beta_table[t][i] = beta_table[t][i] + A_table[i][j] * B_table[j][Ot1] * beta_table[t+1][j]

    # beta_00 = P_O = alpha_(T+1)N
    beta_table[0][0] = P_O

    # compute sum_1-(T-1) (Xi_ij)
    for t in xrange(1, T):
        Ot1 = dict_words[sent[t][0]]
        for i in xrange(1, num_tags):
            for j in xrange(1, num_tags):
                XI_table[i][j] = XI_table[i][j] + (alpha_table[t][i] * A_table[i][j] * B_table[j][Ot1] * beta_table[t+1][j])/P_O

    # compute Xi_0i
    Ot1 = dict_words[sent[0][0]]
    for i in xrange(1, num_tags):
        XI_table[0][i] = XI_table[0][i] + 1 * A_table[0][i] * B_table[i][Ot1] * beta_table[1][i] / P_O

    # compute Xi_i(N)
    for i in xrange(1, num_tags):
        XI_table[i][num_tags] = XI_table[i][num_tags] + alpha_table[T][i] * A_table[i][num_tags] * 1 * 1 / P_O

    # compute gamma_tj, no need to think about alpha_TN
    for t in xrange(1, T+1):
        for j in xrange(1, num_tags):
            Ok = dict_words[sent[t-1][0]]
            GAMMA_table[j][Ok] =GAMMA_table[j][Ok] + alpha_table[t][j] * beta_table[t][j] / P_O

    # compute sum_1w(gamma_wj)
    for j in xrange(1, num_tags):
        for w in xrange(1, num_words):
            GAMMA_table[j][0] = GAMMA_table[j][0] + GAMMA_table[j][w]

    # compute sum_1-(T-1) (sum_1-N (Xi_ij))
    for i in xrange(0, num_tags):
        for j in xrange(1, num_tags+1):
            XI_table[i][0] = XI_table[i][0] + XI_table[i][j]        

    # compute new Aij:
    for i in xrange(0, num_tags):
        for j in xrange(1, num_tags+1):
            new_aij = XI_table[i][j] / XI_table[i][0]
            if abs(new_aij - A_table[i][j]) > epsilon:
                converged = False
            A_table[i][j] = new_aij

    # compute new Bi(w)
    for w in xrange(1, num_words):
        for i in xrange(1, num_tags):
            new_biw = GAMMA_table[i][w] / GAMMA_table[i][0]
            if abs(new_biw - B_table[j][w]) > epsilon:
                converged = False
            B_table[j][w] = new_biw

    converged = True
    print 'ONE ITERATION'
    # test on the test set for every iteration??? (for every sentence?)
    # invoke viterbi algorithm