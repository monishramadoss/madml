import unittest
import madml
import madml.nn as nn
from madml.test import load, mnist_model, train_loop


class Test_models(unittest.TestCase):
    def test_mnist(self):
        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 1, 1, 28, 28))
        x1 = x1.reshape((-1, 1, 1, 1, 28, 28))
        train_loop(x, y, mnist_model())
        self.assertTrue("mnist works")
    
    def test_rnn(self):
        class rnn_seq(nn.Module):
            def __init__(self):
                super(rnn_seq, self).__init__()
                self.rnn = nn.RNN(28*28, 64, 2, batch_first=True, bidirectional=True)
            def forward(self, x):
                rnn_output = self.rnn(x)    

        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, rnn_seq())

    def test_lstm(self):
        class lstm_seq(nn.Module):
            def __init__(self):
                super(lstm_seq, self).__init__()
                self.rnn = nn.LSTM(28*28, 64, 2, batch_first=True, bidirectional=True)
            def forward(self, x):
                rnn_output = self.rnn(x)    


        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, lstm_seq())

    def test_gru(self):
        class gru_seq(nn.Module):
            def __init__(self):
                super(gru_seq, self).__init__()
                self.rnn = nn.GRU(28*28, 64, 2, batch_first=True,  bidirectional=True)
            def forward(self, x):
                rnn_output = self.rnn(x)    


        x, y, x1, y1 = load()
        x = x.reshape((-1, 32, 28*28))
        x1 = x1.reshape((-1, 1, 28*28))
        train_loop(x, y, gru_seq())
        
    
if __name__ == '__main__':
    unittest.main()
