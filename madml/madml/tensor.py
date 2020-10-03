#import backend
import numpy as np

class tensor:
	def __init__(self, shape:list(), data=None, dtype=float, init=False):
		self.shape = shape
		if not init:
			self.data = np.zeros(shape) if data is None else data

	@classmethod
	def reshape(cls, shape):
		cls.data.reshape(shape)

	@classmethod
	def __eq__(cls, value):
		return tensor(cls.shape, data=cls.data == value)

	@classmethod
	def __ne__(cls, value):
		return tensor(cls.shape, data=cls.data != value)

	@classmethod
	def __le__(cls, value):
		return tensor(cls.shape, data=cls.data < value)

	@classmethod
	def __gt__(cls, value):
		return tensor(cls.shape, data=cls.data > value)

	@classmethod
	def __add__(cls, value):
		return tensor(cls.shape, data=cls.data + value)

	@classmethod
	def __sub__(cls, value):
		return tensor(cls.shape, data=cls.data - value)

	@classmethod
	def __mul__(cls, value):
		return tensor(cls.shape, data=cls.data * value)

	@classmethod
	def __div__(cls, value):
		return tensor(cls.shape, data=cls.data * value)