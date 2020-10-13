import unittest

import numpy as np
from urllib import request
import gzip
import pickle
from tqdm import tqdm
import os

import madml
import madml.nn as nn

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

if not os.path.exists('./data'):
    os.makedirs('./data')

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], './data/'+ name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open('./data/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
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

class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
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

class rnn_seq(nn.Module):
    def __init__(self):
        super(rnn_seq, self).__init__()
        self.rnn = nn.RNN(28*28, 64, 2, batch_first=True, bidirectional=True)
    def forward(self, x):
        rnn_output = self.rnn(x)    
    

class lstm_seq(nn.Module):
    def __init__(self):
        super(lstm_seq, self).__init__()
        self.rnn = nn.LSTM(28*28, 64, 2, batch_first=True, bidirectional=True)
    def forward(self, x):
        rnn_output = self.rnn(x)    
    

class gru_seq(nn.Module):
    def __init__(self):
        super(gru_seq, self).__init__()
        self.rnn = nn.GRU(28*28, 64, 2, batch_first=True,  bidirectional=True)
    def forward(self, x):
        rnn_output = self.rnn(x)    
    

def train_loop(x, y, model):
    for i in tqdm(range(x.shape[0])):
        pred = model(x[i,...].astype(np.float32))
        break


class Test_models(unittest.TestCase):
    def test_mnist(self):
        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 1, 1, 28, 28))
        x1 = x1.reshape((-1, 1, 1, 1, 28, 28))
        train_loop(x, y, mnist_net())
        self.assertTrue("mnist works")
    
    def test_rnn(self):
        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, rnn_seq())

    def test_lstm(self):
        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, lstm_seq())

    def test_gru(self):
        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, gru_seq())
        
    
if __name__ == '__main__':
    unittest.main()
