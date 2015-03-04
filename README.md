# CSE5525HW2
NLP Homeowork 2 - HMM and Applications

Authors: Yiran Luo, Dingkang Wang, Zhaoyu Duan, Kwan-wen Lo (Group 1)
Last update: Mar 4, 2015

# Abstract: What are in this package in a nutshell

Step 1 ~ Step 3: main3.py in the root folder contains the probability tables, and a simple test on Viterbi algorithm. Basically it's a demo for Step 1 ~ Step 3
Step 4 ~ Step 6 :implementations are in their corresponding subfolders.
Extra Credit folder contains both written answers and source codes for respecting extra credit questions.

# Documentation Part

#step1 Training sets setup

This is step is provided in the tutorial. 
full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
which in Python returns a list of lists of (tag, word) pairs

#step2 Propagating the probabilities

For demonstration purposes, we take propagating full_training_set as the example. That for training set 1 follow the identical procedure.

First of all, we can acquire the set of all unique words and the set of all unique tags:

full_tags = [tag for (tag, word) in full_training_set_words]
full_words = [word for (tag, word) in full_training_set_words]    
full_tag_set = set(full_tags)
full_word_set = set(full_words)

Then we need to process each sentence in the training set, by adding a start tag and an end tag, and concatenate all the sentences into a single one-dimension list.

for sent in full_training_set:
    full_training_set_words.append(('<s>','<s>'))
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
    full_training_set_words.append(('</s>','</s>'))

Using the one-dimension list, it will be convenient to build the emission frequency distribution by using nltk.ConditionalFreqDist.

full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)

The transition frequency distribution can be built directly with the full tag sequence in bi-gram.

full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))

The transition probability of P(tag2 | tag1) will be N(tag2, tag1) / SUM(N(each tag, tag1))

for tag_1 in full_tag_set:
    if tag_1 != '</s>':
        full_num = 0
        for tag_2 in full_tag_set:
            if tag_2 != '<s>' and (tag_1 != '<s>' or tag_2 != '</s>'):
                if full_cfd_tags[tag_1][tag_2] == 0:
                    full_cfd_tags[tag_1][tag_2] = 1
                full_num = full_num + full_cfd_tags[tag_1][tag_2]
        for tag_2 in full_tag_set:
                A_full_table[dict_tags[tag_1]][dict_tags[tag_2]] = 
float(full_cfd_tags[tag_1][tag_2])/float(full_num)

Notice in order to avoid zero probability/log zero in future, each zero (tag, tag) frequency is replaced with 1 for smoothing purposes.

Similarly the emission probability table will be propagated using the collected frequencies for each (word, tag) pair.

#step3 Viterbi

The original Viterbi Algorithm requires 5 arguments which are states, observations, initial probabilities, transition probabilities and emission probabilities. In our version, we add two additional parameters which are dictionary of tags and dictionary of words since we constructed our probability table with lists and dictionary rather than the default method in nltk api(.prob) which is time-consuming.
In our application, the observations are the words of a sentence and states are tags.
We can apply the transition probabilities and emission probabilities as following.

emit_p[dict_tag[State]][dict_word[Observation]]
trans_p[dict_tag[Statei]][dict_tag[Statej]]

We assign the initial probabilities of <s> to 1 and the other tags for 0.00000000000000000001
init_table: The initial probabilities
init_table[<s>] = 1
init_table[State] =  0.00000000000000000001

A_full_table: The transmission probabilities
B_full_table:The emission probabilities

viterbi(dict_tags, dict_words, test_obs, full_tag_set, init_table, A_full_table, B_full_table )
The output will return a list which is the most likely path of states.
Eg. ['<s>', u'CD', u'CD', u',', u'CD', u'NNS', u'JJ', u'MD', u'VB', u'DT', u'NN', u'IN', u'DT', u'JJ', u'NN', u'NNP', u'CD', u'.', '</s>']

#step4

#step5

#step6 Forward-backward

You can run the python code on ipython notebook. The source files are located in the “step 6” folder.

First, we will use the probability tables generated from training set1 to initialize the transition table and emission table.

Then, for each input sentences in training set2, we will first compute the alpha and beta values of each node, then add the corresponding values into the Xi and Gamma table based on the equations on the textbook.

After one iteration, i.e., from the first sentence to the last sentence, we will get new transition table and emission table using the Xi and Gamma table.

The algorithm will converge when the sum of the changed probability (error_value) is less than 0.01 (epsilon). After convergence, we can do smoothing.

At last, we can test our new probability tables by invoking the function in step 4.

Note:
Because there are too many sentences in the training set2, it runs pretty slow. So I run the code over the first ten sentences in training set2, it converged, but the error rate over test set is pretty high, and do not decrease, since we train our probability tables based on such a small dataset, there will be some unseen bigrams in test set.

I cannot do smoothing for each iteration, because the error_value will not be small enough, and it won't converge, and it will make the program slower...

For each sentence, the program will output a P_O, which means the probability of the occurrence of that sentence, and it will also output 'ONE ITERATION' after one iteration, and the error_value.

At last, we can test our new probability tables by invoking the function in step 3, 4, and 5.