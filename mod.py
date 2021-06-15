import madml
import madml.nn as nn
import madml.optimizer as optimizer

import numpy as np

from sklearn.datasets import load_digits

def test_linear():
    a = np.random.ranf([3, 5]).astype(np.float32)
    t1 = madml.tensor(a)
    module = nn.linear(5, 5, use_gpu=True)

    t2 = module._forward_cpu(t1)
    y = t2.host_data

    t3 = module._forward_gpu(t1)
    y_hat = t3.download()

    t1.gradient.host_data = a

    print(y_hat == y)

    input()

def test_convolution():
    kernel_shape = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    x = np.array([[[[0., 1., 2., 3., 4.],
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(np.float32)
    y_with_padding = np.array([[[12., 21., 27., 33., 24.],
                                [33., 54., 63., 72., 51.],
                                [63., 99., 108., 117., 81.],
                                [93., 144., 153., 162., 111.],
                                [72., 111., 117., 123., 84.]]]).astype(np.float32).reshape([1, 1, 5, 5])

    t1 = madml.tensor(x)

    module = nn.conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')

    t2 = module.forward(t1)
    y = t2.host_data
    module.to(0)
    t3 = module.forward(t1)
    y_hat = t3.download()

    print(y_hat == y)
    input()

def test_maxpool():
    kernel_shape = [2, 2]
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]

    x = np.arange(0, 100).astype(np.float32).reshape([2, 2, 5, 5])
    t1 = madml.tensor(x)
    print(t1, '\n----------------------\n')

    module = nn.maxpool2d(kernel_shape, stride, padding, dilation)
    t3 = module(t1)
    y_hat = t3.host_data
    print(y_hat, '\n\n')#, dx_hat, '\n\n')
    print('---------------------')

    input()

def test_relu():
    x = np.random.uniform(-2, 2, size=81).reshape([9, 9])
    t1 = madml.tensor(x)
    module = nn.relu()
    t3 = module._forward_gpu(t1)
    y_hat = t3.download()
    print(y_hat)
    print()

    t2 = module._forward_cpu(t1)
    y = t2.host_data
    print(y)
    input()

def test_identity():
    class identity_model(nn.Module):
        def __init__(self):
            super(identity_model, self).__init__()
            self.fc1 = nn.linear(32, 32, False)
            self.fc2 = nn.linear(32, 32, False)

        def forward(self, X):
            X = self.fc1(X)
            X = self.fc2(X)
            return X

    model = identity_model()
    print(model.parameters())
    x = np.ones((2, 32))
    t_x = madml.tensor(x)
    t_y = madml.tensor(x.copy())
    loss_fn = nn.mseloss()
    optim = optimizer.adam(model.parameters(), lr=1e-2)

    for i in range(108):
        optim.zero_grad()
        logit = model(t_x)
        loss = loss_fn(logit, t_y)
        loss.backward()
        optim.step()
        print('===', i, logit.shape, loss.host_data, loss_fn.accuracy())
        if i % 10 == 0:
            print(logit.host_data)

def train_loop(model, loss_fn, optim, t_x, t_y, epochs=10, early_break=-1):
    count = 0
    for _ in range(epochs):
        for i in range(t_x.shape[0]):
            optim.zero_grad()
            logit = model(t_x[i])
            loss = loss_fn(logit, t_y[i])
            loss.backward()
            optim.step()
            print('===', count, logit.shape, loss.host_data, loss_fn.accuracy())
            count += 1
            if i % t_x.shape[0] - 1 == 0 and i != 0:
                print('logit [', end=' ')
                for j in range(10):
                    print(logit.host_data[0][j], end='] ' if j == 9 else ', ')
                print(': target [', end=' ')
                for j in range(10):
                    print(t_y[i].host_data[0][j], end=']\n' if j == 9 else ', ')
            if count == early_break:
                return

def test_loop(model, t_x, t_y, early_stop=-1):
    accuracies = list()
    count = 0
    for i in range(t_x.shape[0]):
        logits = model(t_x[i])
        logits = np.argmax(logits.host_data, axis=-1)
        target = np.argmax(t_y[i].host_data)
        accuracies.append(1.0 - (logits - target).mean())
        if count == early_stop:
            break
    return accuracies

def test_mnst_cnn():

    class cnn_mnist_model(nn.Module):
        def __init__(self):
            super(cnn_mnist_model, self).__init__()
            self.conv1 = nn.conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.conv2d(32, 32, 1, stride=2)
            self.conv3 = nn.conv2d(32, 48, 3)
            self.fc1 = nn.linear(48 * 2 * 2, 120) # (599, 192)
            self.fc2 = nn.linear(120, 84)
            self.fc3 = nn.linear(84, 10)

            self.relu1 = nn.relu()
            self.relu2 = nn.relu()
            self.relu3 = nn.relu()
            self.relu4 = nn.relu()

            #self.fc3.to(0)
            #self.fc2.to(0)
            #self.fc1.to(0)

            #self.conv1.to(0)
            #self.conv2.to(0)
            #self.conv3.to(0)

        def forward(self, X):
            X = self.conv1(X)
            X = self.relu1(X)
            X = self.conv2(X)
            X = self.conv3(X)
            X = self.relu2(X)
            X = madml.flatten(X)
            X = self.fc1(X)
            X = self.relu3(X)
            X = self.fc2(X)
            X = self.relu4(X)
            X = self.fc3(X)
            return X

    BATCHSIZE = 599

    x, y, = load_digits(return_X_y=True)
    tx, ty = x[:-100], y[:-100]
    x = x.reshape((-1, BATCHSIZE, 1, 8, 8))
    y = y.reshape((-1, BATCHSIZE, 1))
    tx = tx.reshape((-1, 1, 1, 8, 8))
    ty = ty.reshape((-1, 1, 1))

    model = cnn_mnist_model()
    t_x = madml.tensor(x / 1.)
    t_y = madml.tensor(y).onehot(label_count=10)
    loss_fn = nn.mseloss()
    #loss_fn = nn.crossentropyloss(with_logit=True)
    optim = optimizer.adam(model.parameters(), lr=1e-3)
    train_loop(model, loss_fn, optim, t_x, t_y, epochs=30)

    #test_x = madml.tensor(tx / 1.)
    #test_y = madml.tensor(ty)
    #acc = test_loop(model, test_x, test_y)
    #print(sum(acc) / len(acc))

if __name__ == "__main__":
    # test_convolution()
    test_maxpool()
    # test_mnst_cnn()
    # test_identity()