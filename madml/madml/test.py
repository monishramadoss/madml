from madml.optimizer import Optimizer
import numpy as np
from urllib import request
import gzip
import pickle
import os
from tqdm import tqdm

import madml
import madml.nn as nn
import madml.optimizer as optimizer

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
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 46, 3)
        self.fc1 = nn.Linear(46 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x) # 32 x 28 x 28
        x = self.relu1(x)
        x = self.pool(x) # 32 x 14 x 14
        x = self.conv2(x) # 46 x 12 x 12
        x = self.relu2(x)
        x = x.reshape((bs, -1))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def train_loop(x, y, model):
    criterion = nn.CrossEntropyLoss()
    optim = optimizer.SGD(model.parameters(), lr=0.001, momentum=0.9)
