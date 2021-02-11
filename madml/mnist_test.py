import numpy as np

import madml
import madml.nn as nn
import madml.optimizer as optimizer

np.random.seed(seed=1024)


class identity_model(nn.Module):
    def __init__(self):
        super(identity_model, self).__init__()
        self.fc1 = nn.Linear(32, 32)

    def forward(self, x):
        x = self.fc1(x)
        return x


def identity_train_loop(model_class=identity_model):
    model = model_class()
    x = np.ones((2, 32))
    t_x = madml.tensor(x)
    t_y = madml.tensor(x.copy())
    loss_fn = nn.MSELoss()
    optim = optimizer.Adam(model.parameters(), lr=1e-2)
    logit = None

    for _ in range(100):
        optim.zero_grad()
        logit = model(t_x)
        loss = loss_fn(logit, t_y)
        loss.backward()
        optim.step()
        print(logit.shape, loss.host_data, loss_fn.accuracy())
    print(logit.host_data)


if __name__ == '__main__':
    identity_train_loop()
