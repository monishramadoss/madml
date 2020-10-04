import backend
import unittest


class Test_test2(unittest.TestCase):
    def test_A(self):
        with Exception as context:
            backend.test_memory()	    
            self.assertTrue('This is broken' in context.exception)


if __name__ == '__main__':
    unittest.main()