import numpy as np
dataset = np.arange(0, 200)
dataset = np.reshape(dataset, (100, 2))

batch_size = 3
indices = [i for i in range(dataset.shape[0])]

times = dataset.shape[0]//batch_size
if dataset.shape[0]%batch_size != 0:
    times += 1
#times = 34
np.random.shuffle(indices)

for i in range(times):
    index_i = indices[batch_size*i:batch_size*(i + 1)]
    data_i = dataset[index_i]
    print(data_i.shape);input()