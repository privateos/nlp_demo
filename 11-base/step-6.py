import utils
import torch
import torch.nn as nn
import torch.optim as optim
X_train, Y_train, X_test, Y_test = utils.get_diabetes_dataset()

epochs = 20
batch_size = 3
shuffle_train = True
shuffle_test = False
learning_rate = 0.001

in_features = X_train.shape[1]
out_features = Y_train.shape[1]
model = utils.MyLinear(in_features, out_features)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = utils.MyMSELoss()

for epoch in range(epochs):
    train_iter = utils.get_batch_data(X_train, Y_train, batch_size=batch_size, shuffle=shuffle_train)
    train_loss = 0
    train_batches = 0
    model.train()
    for x, y in train_iter:
        x_torch = torch.from_numpy(x).float()
        y_torch = torch.from_numpy(y).float()
        predict = model(x_torch)
        loss = loss_fn(predict, y_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1
    with torch.no_grad():
        test_iter = utils.get_batch_data(X_test, Y_test, batch_size=batch_size, shuffle=shuffle_test)
        model.eval()
        test_loss = 0.0
        test_batches = 0
        for x, y, in test_iter:
            torch_x = torch.from_numpy(x).float()
            torch_y = torch.from_numpy(y).float()
            predict = model(torch_x)

            loss = loss_fn(predict, torch_y)

            test_loss += loss.item()
            test_batches += 1

    # print(f'epoch={epoch},train_loss = {train_loss/train_batches},test_loss={test_loss/test_batches}')
    message = 'epoch={0:>3},train_loss = {1:>6.4},test_loss={2:>6.4}'
    print(message.format(epoch, train_loss/train_batches, test_loss/test_batches))    


