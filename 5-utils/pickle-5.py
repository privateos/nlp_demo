import pickle as pkl
vocab_path = 'vocab.pkl'
# a = {'源': 0, '珊': 1, '爱': 2, '我': 3, '是': 4}
# f = open(vocab_path, 'wb')
# pkl.dump(a, f)


b = pkl.load(open(vocab_path, 'rb'))#read binary
print(b)