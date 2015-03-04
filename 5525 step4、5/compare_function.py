def compare_sets(set1,set2,indicator): 
    """when indicator==words_match_rate,this function will return the rate that words are tagged correctly
when indicator== sents_match_rate, this function will return the rate sentences are tagged correctly"""
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
 
