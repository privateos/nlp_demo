from audioop import bias
import utils
import torch.optim as optim
import torch.nn as nn
import torch

X_train, Y_train, X_test, Y_test = utils.get_diabetes_dataset()

in_features = X_train.shape[1]
out_features = Y_train.shape[1]
# my_linear = utils.MyLinear(in_features, out_features)
my_linear = nn.Linear(in_features, out_features)#X@W + b
loss_fn = nn.MSELoss()

epochs = 10
batch_size = 3
shuffle_train = True
shuffle_test = False
lr = 0.001

optimizer = optim.Adam(my_linear.parameters(), lr)

for epoch in range(epochs):
    train_iter = utils.get_batch_data(X_train, Y_train, batch_size, shuffle_train)
    train_loss = 0.0
    train_batch = 0
    my_linear.train()
    for x, y in train_iter:
        #x.shape = (batch_size, in_features)
        x_torch = torch.from_numpy(x).float()
        y_torch = torch.from_numpy(y).float()
        #predict.shape = (batch_size, out_features)
        predict = my_linear(x_torch)

        loss = loss_fn(predict, y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batch += 1
    with torch.no_grad():
        my_linear.eval()#evaluate
        test_iter = utils.get_batch_data(X_test, Y_test, batch_size, shuffle_test)
        test_loss = 0.0
        test_batch = 0
        for x, y in test_iter:
            x_torch = torch.from_numpy(x).float()
            y_torch = torch.from_numpy(y).float()

            predict = my_linear(x_torch)
            loss = loss_fn(predict, y_torch)
            test_loss += loss.item()
            test_batch += 1
        
    message = 'epoch={0:>3}, train_loss={1:>7.4}, test_loss={2:>7.4}'
    print(message.format(epoch, train_loss/train_batch, test_loss/test_batch))
    # print(f'train_loss={train_loss/train_batch}')



        