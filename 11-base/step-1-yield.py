# import numpy as np

# def get_batch_data(X, Y, batch_size=1, shuffle=False):
#     N = X.shape[0]
#     num_batch = N//batch_size
#     if N%batch_size != 0:
#         num_batch += 1
    
#     indices = [i for i in range(N)]
#     if shuffle:
#         np.random.shuffle(indices)
    
#     for i in range(num_batch):
#         start_index = batch_size*i
#         end_index = batch_size*(i + 1)
#         index_i = indices[start_index:end_index]

#         x_i = X[index_i]
#         y_i = Y[index_i]
#         yield x_i, y_i



import utils
import numpy as np
N = 10
batch_size = 3
shuffle = True
X = np.random.randn(N, 3)
Y = np.random.randn(N, 1)

for x, y in utils.get_batch_data(X, Y, batch_size=batch_size, shuffle=shuffle):
    print(x, y);input()

