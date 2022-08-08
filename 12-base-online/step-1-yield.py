import numpy as np

#X.shape = (N, in_features)
#Y.shape = (N, out_features)
def get_batch_data(X, Y, batch_size=1, shuffle=False):
    N = X.shape[0]
    num_batch = N//batch_size
    if N%batch_size != 0:
        num_batch += 1
    
    indices = [i for i in range(N)]
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(num_batch):
        start_index = batch_size*i
        end_index = batch_size*(i + 1)
        indices_i = indices[start_index:end_index]
        X_i = X[indices_i]
        Y_i = Y[indices_i]
        yield X_i, Y_i

np.random.seed(0)
X = np.random.randn(5, 3)
Y = np.random.randn(5, 2)
batch_size = 2
print('X = ',X)
print('Y = ', Y)
dataset = get_batch_data(X, Y, batch_size=batch_size)
for x, y in dataset:
    print(x)
    print(y)
    input()
