import backend
import unittest


class Test_backend(unittest.TestCase):
    def test_memory(self):
        backend.test_memory()	    
        self.assertTrue('This works')


if __name__ == '__main__':
    unittest.main()