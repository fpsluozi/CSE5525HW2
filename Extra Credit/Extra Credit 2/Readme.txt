Run the python code in ipython notebook, and it will generate a data file in the folder of notebook.

Put the generated data file to your default folder, and run the following two SRILM commands, then you can find the probability file in data.2gram which used backoff.

ngram-count -order 2 -text data -write data.2counts

ngram-count -order 2 -read data.2counts -lm data.2gram

Or you can see the generated data, data.2counts and data.2gram files directly, the final results with backoff is in file data.2gram.