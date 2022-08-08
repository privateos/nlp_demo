f = [
    ['我','是','一','头','大','肥','猪'],
    ['源','源','爱','珊','珊'],
    ['珊','珊','爱','源','源']
]
vocab_dic = {}
for line in f:
    for word in line:
        vocab_dic[word] = vocab_dic.get(word, 0) + 1
print(vocab_dic);input()

########################################################################
# vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
min_freq = 1
max_size = 5
sl = [_ for _ in vocab_dic.items() if _[1] >= min_freq]
print(sl);input()
vocab_list = sorted(sl, key=lambda x: x[1], reverse=True)[:max_size]
print(vocab_list);input()


#######################################################################
vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
print(vocab_dic)
