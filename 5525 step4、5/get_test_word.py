def get_test_word(set):
	word_set=[]
	sent_set=[]
	set_length=len(set)
	sentence_length=0
	for i in range(0,set_length):
		sentence_length=len(set[i])
		for j in range(0,sentence_length):
			sent_set.append(set[i][j][0])
		word_set.append(sent_set)
		sent_set=[]
	return word_set
