You can run the python code on ipython notebook.



First, we will use the probability tables generated from training set1 to initialize the transition table and emission table.

Then, for each input sentences in training set2, we will first compute the alpha and beta values of each node, then add the corresponding values into the Xi and Gamma table based on the equations on the textbook.

After one iteration, i.e., from the first sentence to the last sentence, we will get new transition table and emission table using the Xi and Gamma table. 

The algorithm will converge when the sum of the changed probability (error_value) is less than 0.01 (epsilon). After convergence, we can do smoothing.

At last, we can test our new probability tables by invoking the function in step 3, 4, and 5.



Note:
Because there are too many sentences in the training set2, it runs pretty slow. So I run the code over the first ten sentences in training set2, it converged, but the error rate over test set is pretty high, and do not decrease, since we train our probability tables based on such a small dataset, there will be some unseen bigrams in test set.

I cannot do smoothing for each iteration, because the error_value will not be small enough, and it won't converge, and it will make the program slower...

For each sentence, the program will output a P_O, which means the probability of the occurance of that sentence, and it will also output 'ONE ITERATION' after one iteration, and the error_value.