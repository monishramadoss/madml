import unittest
import backend

#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#        self.relu1 = nn.ReLU()
#        self.relu2 = nn.ReLU()
#        self.relu3 = nn.ReLU()
#        self.relu4 = nn.ReLU()

#    def forward(self, x):
#        x = self.pool(self.relu1(self.conv1(x)))
#        x = self.pool(self.relu2(self.conv2(x)))
#        x = x.reshape((-1, 16 * 5 * 5))        
#        x = self.relu3(self.fc1(x))
#        x = self.relu4(self.fc2(x))
#        x = self.fc3(x)
#        return x


class Test_test_1(unittest.TestCase):
    def test_A(self):
        import madml
        self.fail("Not implemented")

if __name__ == '__main__':
    unittest.main()
