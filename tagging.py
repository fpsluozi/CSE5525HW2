from combine_tag_word import combine_tag_word
def tagging(set):
	global full_tags
	global init_table
	global full_cpd_tags
	global full_cpd_word_tag
	tagset=[]
	for i in range (0,len(set)):
		tagset.append(viterbi(set[i], set(full_tags), init_table, full_cpd_tags, full_cpd_word_tag))
	return combine_tag_word(set,tagset)