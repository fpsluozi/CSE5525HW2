The first HMM_tagger is a word tagger, which means the input is a sequence of words, and the output is the predicted tags according to the transition and emission prob tables. The output is consist of a origin sentences, a right tag sequence, and the predicted tag sequence.

The second HMM_tagger is a chunker, which means the input is a sequence of tags, and the output is the predicted BOI symbols according to the prob tables. Similarly, the output is consist of a origin (tag, chunk) sequence, the right BOI output, and the predicted BOI output.

You can run both of them in ipython notebook, and you can change the index of the testing sentence. 

You can see more notes in the python files.