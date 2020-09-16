import backend
import unittest

class Test_backend(unittest.TestCase):
	def test_memory(self):
		backend.test_memory()
	def test_transforms(self):
		backend.test_trans()
	def test_math(self):
		backend.test_math()

if __name__ == '__main__':
	unittest.main()