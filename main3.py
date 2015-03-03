# CSE5525 NLP Homework 2 Group 1
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
<<<<<<< HEAD
            

#step3 viterbi

# The init_table is the initial probabilities and enforces every sentence to start with <s>
init_table = {}
for tag in full_tag_set:
    init_table[tag] = 0.00000000000000000001
init_table['<s>'] = 1.0

# test_obs is a sentence to test viterbi
test_obs = ['<s>', 'Pierre', 'Viken', ',' , '61' , "years", "old", "will", "join", "the", "board", "as", "a", "nonexecutive", "director", "Nov.", "29","."]

#run viterbi
print viterbi(dict_tags, dict_words, test_obs, full_tags, init_table, A_full_table, B_full_table )
=======




#step 6

epsilon = 0.01

converged = False

while not converged:
    
    converged = True
    
    # running forward-backward training
    # for every sentence in full_training_set as one iteration.

    # create Xi, Gamma table
    XI_table = numpy.zeros((num_tags, num_tags), dtype='double')
    GAMMA_table = numpy.zeros((num_tags, num_words), dtype='double')
    
    for sent in training_set2:
        
        # get alpha
        T = len(sent)
        alpha_table = numpy.zeros((T + 2, num_tags), dtype='double') 

        # alpha_00 = 1
        alpha_table[0][0] = 1

        # intialize alpha_1j = a_0j * bj(W1)
        for i in xrange(1, num_tags - 1):
            j = dict_words[sent[0][0]] # Word1
            alpha_table[1][i] = A_set1_table[0][i] * B_set1_table[i][j]
            # print A_set1_table[0][i], " ", B_set1_table[i][j]
            
        # compute alpha_tj
        for t in xrange(2, T+1):
            Ot = dict_words[sent[t-1][0]]
            for i in xrange(1, num_tags - 1):
                for j in xrange(1, num_tags - 1):
                    alpha_table[t][i] = alpha_table[t][i] + alpha_table[t-1][j] * A_set1_table[j][i] #* B_set1_table[i][Ot]
                alpha_table[t][i] = alpha_table[t][i] * B_set1_table[i][Ot]

        # compute alpha_(T+1)N
        for i in xrange(1, num_tags - 1):
            alpha_table[T+1][num_tags - 1] = alpha_table[T+1][num_tags - 1] + alpha_table[T][i] * A_set1_table[i][num_tags - 1]

        # compute P(O|lamda) = aplha_(T+1)N
        #P_O = 0
        #for i in xrange(1, num_tags):
        #   P_O = P_O + alpha[T][i] * END_table[i]
        P_O = alpha_table[T+1][num_tags - 1]
        print "P_O = ", P_O

        # get beta
        beta_table = numpy.zeros((T + 2, num_tags), dtype='double')

        # beta_(T+1)N = 1
        beta_table[T+1][num_tags - 1] = 1

        #initialize beta_Ti = aiN
        for i in xrange(1, num_tags - 1):
            beta_table[T][i] = A_set1_table[i][num_tags - 1]
            
        # compute beta_ti
        for t in xrange(T-1, 0, -1):
            Ot1 = dict_words[sent[t][0]]
            for i in xrange(1, num_tags - 1):
                for j in xrange(1, num_tags - 1):
                    beta_table[t][i] = beta_table[t][i] + A_set1_table[i][j] * beta_table[t+1][j] * B_set1_table[j][Ot1]
                # print t, " ", i, " ", beta_table[t][i]
            
        # beta_00 = P_O = alpha_(T+1)N
        beta_table[0][0] = P_O

        # compute sum_1-(T-1) (Xi_ij)
        for t in xrange(1, T):
            Ot1 = dict_words[sent[t][0]]
            for i in xrange(1, num_tags - 1):
                for j in xrange(1, num_tags - 1):
                    XI_table[i][j] = XI_table[i][j] + (alpha_table[t][i] * A_set1_table[i][j] * B_set1_table[j][Ot1] * beta_table[t+1][j])/P_O
            
        # compute Xi_0i
        Ot1 = dict_words[sent[0][0]]
        for i in xrange(1, num_tags - 1):
            XI_table[0][i] = XI_table[0][i] + 1 * A_set1_table[0][i] * B_set1_table[i][Ot1] * beta_table[1][i] / P_O
            # print i, " ", A_set1_table[0][i], " ", B_set1_table[i][Ot1], " ", beta_table[1][i]
            
        # compute Xi_i(N)
        for i in xrange(1, num_tags - 1):
            XI_table[i][num_tags - 1] = XI_table[i][num_tags - 1] + alpha_table[T][i] * A_set1_table[i][num_tags - 1] * 1 * 1 / P_O

        # compute gamma_tj, no need to think about alpha_TN
        for t in xrange(1, T + 1):
            Ok = dict_words[sent[t-1][0]]
            for j in xrange(1, num_tags - 1):
                GAMMA_table[j][Ok] = GAMMA_table[j][Ok] + alpha_table[t][j] * beta_table[t][j] / P_O
            #print sent[t-1][0], " ", GAMMA_table[j][Ok]
        
        
    # compute sum_1w(gamma_wj)
    for j in xrange(1, num_tags - 1):
        for w in xrange(1, num_words - 1):
            GAMMA_table[j][0] = GAMMA_table[j][0] + GAMMA_table[j][w]

    # compute sum_1-(T-1) (sum_1-N (Xi_ij))
    for i in xrange(0, num_tags - 1):
        for j in xrange(1, num_tags):
            XI_table[i][0] = XI_table[i][0] + XI_table[i][j]  
    
    
    error = 0.0
    # compute new Aij:
    for i in xrange(0, num_tags - 1):
        for j in xrange(1, num_tags):
            new_aij = XI_table[i][j] / XI_table[i][0]
            if abs(new_aij - A_set1_table[i][j]) > epsilon:
                error = error + abs(new_aij - A_set1_table[i][j])
            A_set1_table[i][j] = new_aij
            
    # compute new Bi(w)
    # for w in xrange(1, num_words - 1):
    for pair in sent:
        w = dict_words[pair[0]]
        for i in xrange(1, num_tags - 1):
            new_biw = GAMMA_table[i][w] / GAMMA_table[i][0]
            if abs(new_biw - B_set1_table[i][w]) > epsilon:
                error = error + abs(new_biw - B_set1_table[i][w])
            B_set1_table[i][w] = new_biw
    
    if error > epsilon:
        converged = False
        
    print error
    print 'ONE ITERATION'
    # test on the test set for every iteration??? (for every sentence?)
    # invoke viterbi algorithm
    
>>>>>>> origin/master

