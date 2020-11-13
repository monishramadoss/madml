
import madml.test as test



train_x, train_y, test_x, test_y = test.load()
test.train_loop(train_x, train_y)