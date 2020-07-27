import numpy as np
from urllib import request
import gzip
import pickle
import os

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
    print('training_images:', mnist["training_images"].shape)
    print('training_labels:', mnist["training_labels"].shape)
    print('test_images', mnist["test_images"].shape)
    print('test_labels', mnist["test_labels"].shape)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]



init()
tr_x, tr_y, te_x, te_y = load()

for x, y in zip(tr_x, tr_y):
    pass

init()
tr_x, tr_y, te_x, te_y = load()

for x, y in zip(tr_x, tr_y):
    pass


