import numpy as np
from urllib import request
import gzip
import pickle
from tqdm import tqdm
import os

import madml
import madml.nn as nn
from madml import tensor
filename = [["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]]

if not os.path.exists('./data'):
    os.makedirs('./data')

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], './data/' + name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open('./data/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28 * 28)
    for name in filename[-2:]:
        with gzip.open('./data/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("./data/mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    if not os.path.exists('./data/mnist.pkl'):
        download_mnist()
        save_mnist()

def load():
    with open("./data/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    print('training_images:', mnist["training_images"].shape)
    print('training_labels:', mnist["training_labels"].shape)
    print('test_images', mnist["test_images"].shape)
    print('test_labels', mnist["test_labels"].shape)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 46, 3)
        self.fc1 = nn.Linear(46 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flat = nn.flatten()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x) # 32 x 28 x 28
        x = self.relu1(x)
        x = self.pool(x) # 32 x 14 x 14
        x = self.conv2(x) # 46 x 12 x 12
        x = self.relu2(x)
        x = self.flat(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

x, y, x1, y1 = load()
x = x.reshape((-1, 32, 1, 1, 28, 28))
x1 = x1.reshape((-1, 1, 1, 1, 28, 28))

print(x.shape, x1.shape)
print(x[0,...].shape, x1[0,...].shape)

tx = tensor(x, x.shape).numpy()
ty = tensor(y, y.shape).numpy()
print(tx.shape, ty.shape)
n = Net()

for i in tqdm(range(x.shape[0])):
    y = n(x[i,...].astype(np.float32))
    break

for i in tqdm(range(x.shape[0])):
    y = n(x1[i,...].astype(np.float32))
    break