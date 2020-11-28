import unittest

class Test_Unittesting_Env(unittest.TestCase):
    def test_pass(self):
        self.assertTrue("Passed Test")

if __name__ == '__main__':
    unittest.main()
