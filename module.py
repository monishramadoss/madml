import madml
import madml.nn as nn
import numpy as np

a = np.random.ranf([3, 5]).astype(np.float32)
t1 = madml.tensor(a)

module = nn.Linear(5, 5)

t2 = module.forward_cpu(t1)
y = t2.host_data

t3 = module.forward_gpu(t1)
y_hat = t3.download()
print(y_hat, y)

input()