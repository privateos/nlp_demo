import numpy as np
dataset = np.arange(0, 200)
dataset = np.reshape(dataset, (100, 2))

batch_size = 3
shuffle = True


def get_batch_data(dataset, batch_size=1, shuffle=False):
    times = dataset.shape[0]//batch_size
    if dataset.shape[0]%batch_size != 0:
        times += 1
    
    indices = [i for i in range(dataset.shape[0])]
    if shuffle:
        np.random.shuffle(indices)
    for i in range(times):
        index_i = indices[batch_size*i:batch_size*(i + 1)]
        data_i = dataset[index_i]
        yield data_i

for data in get_batch_data(dataset, batch_size=batch_size, shuffle=shuffle):
    print(data);input()