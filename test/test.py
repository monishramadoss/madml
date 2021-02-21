import gzip
import os
import pickle
import unittest
from urllib import request

import numpy as np
import matplotlib.pyplot as plt

BATCHSIZE = 100

class TestImports(unittest.TestCase):
    def test_madml(self):
        try:
            import madml
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_backend(self):
        try:
            import backend
            self.assertTrue(True)
        except:
            self.assertTrue(False)
    
    def test_vknn(self):
        try:
            import vknn
            self.assertTrue(True)
        except:
            self.assertTrue(False)
        


class TestModules(unittest.TestCase):
    def test_tensor(self):
        
        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        t1 = madml.tensor(x)
        self.assertTrue(t1.shape == list(x.shape))
        self.assertTrue((t1.host_data == x).all())

    def test_conv(self):

        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

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
        module = nn.Conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')
        t2 = module.forward_cpu(t1)
        y = t2.host_data
        self.assertTrue((y == y_with_padding).all())

        padding = [0, 0]
        y_without_padding = np.array([[[[54., 63., 72.],
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(np.float32).reshape([1, 1, 3, 3])
        module2 = nn.Conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')
        t3 = module2.forward_cpu(t1)
        y2 = t3.host_data
        self.assertTrue((y2 == y_without_padding).all())

        dy = np.array([[[[0., 1., 2.],
                         [3., 4., 5.],
                         [6., 7., 8.]]]]).astype(np.float32).reshape([1, 1, 3, 3])
        dx = np.array([[[[0., 1., 3., 3., 2.],
                         [3., 8., 15., 12., 7.],
                         [9., 21., 36., 27., 15.],
                         [9., 20., 33., 24., 13.],
                         [6., 13., 21., 15., 8.]]]]).reshape([1, 1, 5, 5])

        t3.gradient.host_data = dy
        _ = module2.backward_cpu()
        y3 = t1.gradient.host_data
        self.assertTrue((y3 == dx).all())

    def test_maxpool(self):

        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        kernel_shape = [2, 2]
        stride = [1, 1]
        padding = [0, 0]
        dilation = [1, 1]

        x = np.arange(0, 100).astype(np.float32).reshape([2, 2, 5, 5])

        t1 = madml.tensor(x)
        module = nn.MaxPool2d(kernel_shape, stride, padding, dilation)
        t2 = module.forward_cpu(t1)
        y = t2.host_data

        test = x[..., 1:, 1:]
        self.assertTrue((test == y).all())
        t2.gradient.host_data = y
        _x = module.backward_cpu()
        dx = t1.gradient.host_data[..., 1:, 1:]
        self.assertTrue(True)

    def test_crossentropy(self):

        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,))

        t1 = madml.tensor(x)
        target = madml.tensor(labels)
        module = nn.CrossEntropyLoss()

        loss = module.forward_cpu(t1, target)

        dx = module.backward_cpu()
        print(loss.host_data, dx.gradient.host_data)

    def test_relu(self):
        x = np.random.uniform(-2, 2, size=81).reshape([9, 9])
        t1 = madml.tensor(x)
        module = nn.ReLU()
        logit = module.forward_cpu(t1)
        logit.gradient.host_data = x
        y = logit.host_data
        dx = module.backward_cpu().gradient.host_data
        self.assertTrue((np.sum(y) == np.sum(dx)).all())


def load_mnist():
    filename = [["training_images", "train-images-idx3-ubyte.gz"],
                ["test_images", "t10k-images-idx3-ubyte.gz"],
                ["training_labels", "train-labels-idx1-ubyte.gz"],
                ["test_labels", "t10k-labels-idx1-ubyte.gz"]]
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if not os.path.exists('./data/mnist.pkl'):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], './data/' + name[1])
        print("Download complete.")
        mnist = {}
        for name in filename[:2]:
            with gzip.open('./data/' + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open('./data/' + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open("./data/mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")
    with open("./data/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def train_loop(model, loss_fn, optim, t_x, t_y):
    for _ in range(10):
        for i in range(t_x.shape[0]):
            optim.zero_grad()
            logit = model(t_x[i])
            loss = loss_fn(logit, t_y[i])
            loss.backward()
            optim.step()
            print('===', i, logit.shape, loss.host_data, loss_fn.accuracy())
            if i % 20 == 0 and i != 0:
                print('logit [', end=' ')
                for j in range(10):
                    print(logit.host_data[0][j], end='] ' if j == 9 else ', ')
                print(': target [', end=' ')
                for j in range(10):
                    print(t_y[i].host_data[0][j], end=']\n' if j == 9 else ', ')


def test_loop(model, t_x, t_y):
    accuracies = list()
    for i in range(t_x.shape[0]):
        logits = model(t_x[i])
        logits = np.argmax(logits.host_data, axis=-1)
        target = np.argmax(t_y[i].host_data)
        accuracies.append(1.0 - (logits - target).mean())
    return accuracies


class TestModels(unittest.TestCase):
    def test_cnn(self):
        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        class cnn_mnist_model(nn.Module):
            def __init__(self):
                super(cnn_mnist_model, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 48, 3)
                self.fc1 = nn.Linear(48 * 12 * 12, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()
                self.relu3 = nn.ReLU()
                self.relu4 = nn.ReLU()

            def forward(self, X):
                X = self.conv1(X)
                X = self.relu1(X)
                X = self.pool(X)  # 32 x 14 x 14
                X = self.conv2(X)  # 46 x 12 x 12
                X = self.relu2(X)
                X.flatten()
                X = self.fc1(X)
                X = self.relu3(X)
                X = self.fc2(X)
                X = self.relu4(X)
                X = self.fc3(X)
                return X

        x, y, tx, ty = load_mnist()
        x = x.reshape((-1, BATCHSIZE, 1, 28, 28))
        y = y.reshape((-1, BATCHSIZE, 1))
        tx = tx.reshape((-1, 1, 1, 28, 28))
        ty = ty.reshape((-1, 1, 1))

        model = cnn_mnist_model()
        self.assertIsInstance(model, nn.Module)

        t_x = madml.tensor(x / 225.)
        t_y = madml.tensor(y).onehot(label_count=10)
        # loss_fn = nn.MSELoss()
        loss_fn = nn.CrossEntropyLoss(with_logit=True)
        optim = optimizer.Adam(model.parameters(), lr=1e-3)
        train_loop(model, loss_fn, optim, t_x, t_y)

        test_x = madml.tensor(tx / 255.)
        test_y = madml.tensor(ty)
        acc = test_loop(model, test_x, test_y)
        print(sum(acc) / len(acc))

    def test_dnn(self):
        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        class dnn_mnist_model(nn.Module):
            def __init__(self):
                super(dnn_mnist_model, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 256)
                self.fc2 = nn.Linear(256, 10)
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()

            def forward(self, X):
                X = self.fc1(X)
                X = self.relu1(X)
                X = self.fc2(X)
                X = self.relu2(X)
                return X

        x, y, tx, ty = load_mnist()
        x = x.reshape((-1, BATCHSIZE, 28 * 28))
        y = y.reshape((-1, BATCHSIZE, 1))
        tx = tx.reshape((-1, 1, 28 * 28))
        ty = ty.reshape((-1, 1, 1))

        model = dnn_mnist_model()
        self.assertIsInstance(model, nn.Module)
        t_x = madml.tensor(x / 225.)
        t_y = madml.tensor(y).onehot(label_count=10)
        # loss_fn = nn.MSELoss()
        loss_fn = nn.CrossEntropyLoss(with_logit=True)
        optim = optimizer.Adam(model.parameters(), lr=1e-3)
        train_loop(model, loss_fn, optim, t_x, t_y)

        test_x = madml.tensor(tx / 255.)
        test_y = madml.tensor(ty)
        acc = test_loop(model, test_x, test_y)
        print(sum(acc) / len(acc))

    def test_identity(self):
        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        class identity_model(nn.Module):
            def __init__(self):
                super(identity_model, self).__init__()
                self.fc1 = nn.Linear(32, 32)
                self.fc2 = nn.Linear(32, 32)

            def forward(self, X):
                X = self.fc1(X)
                X = self.fc2(X)
                return X

        model = identity_model()
        self.assertIsInstance(model, nn.Module)
        x = np.ones((2, 32))
        t_x = madml.tensor(x)
        t_y = madml.tensor(x.copy())
        loss_fn = nn.MSELoss()
        optim = optimizer.Adam(model.parameters(), lr=1e-2)

        for i in range(100):
            optim.zero_grad()
            logit = model(t_x)
            loss = loss_fn(logit, t_y)
            loss.backward()
            optim.step()
            print('===', i, logit.shape, loss.host_data, loss_fn.accuracy())
        self.assertTrue(loss_fn.accuracy() > 0.9)

    def test_spiral(self):
        import madml
        import madml.nn as nn
        import madml.optimizer as optimizer

        from numpy import pi
        # import matplotlib.pyplot as plt

        N = 400
        theta = np.sqrt(np.random.rand(N)) * 2 * pi  # np.linspace(0,2*pi,100)
        r_a = 2 * theta + pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        x_a = data_a + np.random.randn(N, 2)
        res_a = np.append(x_a, np.zeros((N, 1)), axis=1)

        r_b = -2 * theta - pi
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
        x_b = data_b + np.random.randn(N, 2)
        res_b = np.append(x_b, np.ones((N, 1)), axis=1)

        res = np.append(res_a, res_b, axis=0)
        np.random.shuffle(res)

        ax1 = plt.subplot(121)
        ax1.margins(0.05)

        ax1.scatter(x_a[:, 0], x_a[:, 1])
        ax1.scatter(x_b[:, 0], x_b[:, 1])

        # np.savetxt("result.csv", res[0], delimiter=",", header="x,y,label", comments="", fmt='%.5f')

        class spiral_model(nn.Module):
            def __init__(self):
                super(spiral_model, self).__init__()
                self.fc1 = nn.Linear(2, 16)
                self.fc2 = nn.Linear(16, 16)
                self.fc3 = nn.Linear(16, 2)
                self.tanh1 = nn.tanh()
                self.tanh2 = nn.tanh()
                self.sig = nn.ReLU()

            def forward(self, X):
                X = self.fc1(X)
                X = self.tanh1(X)
                X = self.fc2(X)
                X = self.tanh2(X)
                X = self.fc3(X)
                X = self.sig(X)
                return X

        model = spiral_model()
        self.assertIsInstance(model, nn.Module)
        x = res[..., :-1]
        y = res[..., 2]

        t_x = madml.tensor(x)
        t_y = madml.tensor(y)
        t_y = t_y.onehot(2)
        t_y.reshape([800, 2])

        loss_fn = nn.CrossEntropyLoss(with_logit=True)
        # loss_fn = nn.MSELoss()
        optim = optimizer.Adam(model.parameters(), lr=1e-2)
        logits = None
        for i in range(100):
            optim.zero_grad()
            logit = model(t_x)
            logits = logit.host_data
            loss = loss_fn(logit, t_y)
            loss.backward()
            optim.step()
            print('===', i, logit.shape, loss.host_data)

        logits = np.argmax(logits, axis=-1)

        result = res[:, :-1]
        ax2 = plt.subplot(122)
        ax2.scatter(result[logits == 0.][:, 0], result[logits == 0.][:, 1])
        ax2.scatter(result[logits == 1.][:, 0], result[logits == 1.][:, 1])
        # plt.savefig('input_output.png')

        acc = (logits - y).mean()
        print(1. - acc)
        self.assertTrue(1.0 - acc > 0.9)


if __name__ == '__main__':
    unittest.main()
