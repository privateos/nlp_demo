from ctypes import util
from sklearn.utils import shuffle

from torch import batch_norm
import utils 

X_train, Y_train, X_test, Y_test = utils.get_diabetes_dataset()
batch_size = 3
shuffle = False

for x, y in utils.get_batch_data(X_train, Y_train, batch_size=batch_size, shuffle=shuffle):
    print(x, y);input()

for x, y in utils.get_batch_data(X_test, Y_test, batch_size=batch_size, shuffle=shuffle):
    print(x, y);input()
