from tqdm import tqdm
import pickle as pkl
import os
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


tokenizer = lambda x: [y for y in x]
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
        # for line in f:
            lin = line.strip()
            # print(lin)
            if not lin:
                continue
            content = lin.split('\t')[0]
            # print(lin.split('\t'))
            # print(tokenizer(content));input()
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

file_path = 'train.txt'
tokenizer = lambda x : [y for y in x]
max_size = MAX_VOCAB_SIZE
min_freq = 1
vocab = build_vocab(file_path, tokenizer, max_size, min_freq)



#从这里开始
def load_dataset(path, pad_size=32):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            # print(words_line, int(label), seq_len);input()

            contents.append((words_line, int(label), seq_len))

    return contents 

test_path = 'test.txt'
pad_size = 32
test = load_dataset(test_path, pad_size)
print(test[0])
