# CSE5525 NLP Homework 2 Group 1
import nltk
import numpy
import itertools
from nltk.corpus import treebank
from get_test_word import get_test_word
from viterbi import viterbi
from nltk.corpus import treebank
from get_test_word_firsttime import get_test_word_firsttime
from compare_function import compare_sets
from combine_tag_word import combine_tag_word
from get_head_and_tail import get_head_and_tail

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# Retrieve P(W_i | T_i) and P(T_i| T_i-1)

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
             
# using the codes in step4 and step5 for getting the precision rate
def tagging(set_,tags,word_tag):
    global full_tags
    global init_table
    global full_cpd_tags
    global full_cpd_word_tag
    tagset=[]
    init_table = {}
    for tag in full_tag_set:
        init_table[tag] = 0.00000000000000000001
    init_table['<s>'] = 1.0
    for i in range (0,len(set_)):
        tagset.append(viterbi(dict_tags, dict_words, set_[i], full_tag_set, init_table, tags, word_tag))
    return combine_tag_word(set_,tagset)

def compare_sets(set1,set2,indicator):
    words_count=0
    # the number of words being compared
    sents_pointer=0
    # which sentence is being compared
    words_pointer=0
    #  which word is being compared
    words_match_rate=0
    #the rate words are tagged correctly
    sents_match_rate=0
    #the rate sentences are tagged correctly
    words_tag_match_count_all=0
    # the number of words that are tagged correctly
    sents_tag_match_count=0
    # the number of sentences that are tagged correctly
    words_tag_match_count=0
    # the number of words that are tagged correctly in the sentence being compared
    sents_count=0
    words_count_eachsentence=0
    if len(set1)<len(set2):
        sents_count=len(set1)
    else:
        sents_count=len(set2)
    for sents_pointer in range(0,sents_count):
        if len(set1[sents_pointer])<len(set2[sents_pointer]):
            words_count_eachsentence=len(set1[sents_pointer])
        else:
            words_count_eachsentence=len(set2[sents_pointer])
        for words_pointer in range(0,words_count_eachsentence):
            words_count=words_count+1
            if set1[sents_pointer][words_pointer][1]==set2[sents_pointer][words_pointer][1]and set1[sents_pointer][words_pointer][0]==set2[sents_pointer][words_pointer][0]:
                words_tag_match_count=words_tag_match_count+1
                words_tag_match_count_all=words_tag_match_count_all+1
        if words_tag_match_count==len(set1[sents_pointer]):
            sents_tag_match_count=sents_tag_match_count+1
            words_tag_match_count=0
    words_match_rate=words_tag_match_count_all/float(words_count)
    sents_match_rate=sents_tag_match_count/float(sents_count)
    if indicator=="words_match_rate":
        return words_match_rate
    elif indicator=="sents_match_rate":
        return sents_match_rate

# forward-backward algorithm

epsilon = 0.01
min_b = 1.0

converged = False

while not converged:
    
    converged = True
    
    # running forward-backward training
    # for every sentence in full_training_set as one iteration.

    # create Xi, Gamma table
    XI_table = numpy.zeros((num_tags, num_tags), dtype='double')
    GAMMA_table = numpy.zeros((num_tags, num_words), dtype='double')
    
    for sent in training_set2:
        
    # you can change the size of the training set.
    # for x in xrange(0, 1):
    #    sent = training_set2[x]
        
        # build alpha table
        T = len(sent)
        alpha_table = numpy.zeros((T + 2, num_tags), dtype='double') 

        # alpha_00 = 1
        alpha_table[0][0] = 1

        # intialize alpha_1j = a_0j * bj(W1)
        for i in xrange(1, num_tags - 1):
            j = dict_words[sent[0][0]] # Word1
            alpha_table[1][i] = A_set1_table[0][i] * B_set1_table[i][j]
            
        # compute alpha_tj
        for t in xrange(2, T+1):
            Ot = dict_words[sent[t-1][0]]
            for i in xrange(1, num_tags - 1):
                for j in xrange(1, num_tags - 1):
                    alpha_table[t][i] = alpha_table[t][i] + alpha_table[t-1][j] * A_set1_table[j][i] #* B_set1_table[i][Ot]
                alpha_table[t][i] = alpha_table[t][i] * B_set1_table[i][Ot]
            #print t, " ", i, " ", alpha_table[t][i]
            
        # compute alpha_(T+1)N
        for i in xrange(1, num_tags - 1):
            alpha_table[T+1][num_tags - 1] = alpha_table[T+1][num_tags - 1] + alpha_table[T][i] * A_set1_table[i][num_tags - 1]

        # compute P(O|lamda) = aplha_(T+1)N
        #P_O = 0
        #for i in xrange(1, num_tags):
        #   P_O = P_O + alpha[T][i] * END_table[i]
        
        P_O = alpha_table[T+1][num_tags - 1]
        if P_O == 0.0:
            continue
        
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
    
    # get the minimum values of B probs
    # using this value to do smoothing
    
    min_b = 1.0
    
    # error_value
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
    
# 
# After convergence, smoothing the prob table, mainly the emission prob table.
# And get the error_rate.
# 
# do smoothing for emission table, for there won't zero values in transition table.
for i in xrange(1, num_tags - 1):
    added_value = 1.0
    for w in xrange(1, num_words - 1):
        if B_set1_table[i][w] < min_b:
            B_set1_table[i][w] = min_b
            added_value = added_value + min_b
    for w in xrange(1, num_words - 1):
        B_set1_table[i][w] = B_set1_table[i][w] / added_value
    
# test the new prob tables 
temp=tagging(get_test_word_firsttime(test_set),A_full_table,B_full_table)
# output the precision rate over test set
print compare_sets(temp,get_head_and_tail(test_set),"words_match_rate")
