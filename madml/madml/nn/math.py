from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import  Optional
from .module import Module
from madml import tensor
import madml
import random
import backend

class abs(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(abs, self).__init__(backend.abs(inplace))
		self.inplace = inplace

class ceil(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(ceil, self).__init__(backend.ceil(inplace))
		self.inplace = inplace

class clip(Module):
	__constants__ = ['min_val', 'max_val', 'inplace']
	inplace : bool
	max_val : float
	min_val : float
	def __init__(self, min_val, max_val, inplace: bool=False) -> None:
		super(clip, self).__init__(backend.clip(min_val, max_val, inplace))
		self.min_val = min_val
		self.max_val = max_val
		self.inplace = inplace

class exp(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(exp, self).__init__(backend.exp(inplace))
		self.inplace = inplace

class floor(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(floor, self).__init__(backend.floor(inplace))
		self.inplace = inplace

class ln(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(ln, self).__init__(backend.ln(inplace))
		self.inplace = inplace

class round(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(round, self).__init__(backend.round(inplace))
		self.inplace = inplace

class sqrt(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(sqrt, self).__init__(backend.sqrt(inplace))
		self.inplace = inplace

class acos(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(acos, self).__init__(backend.acos(inplace))
		self.inplace = inplace

class acosh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(acosh, self).__init__(backend.acosh(inplace))
		self.inplace = inplace

class asin(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(asin, self).__init__(backend.asin(inplace))
		self.inplace = inplace

class asinh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(asinh, self).__init__(backend.asinh(inplace))
		self.inplace = inplace

class atan(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(atan, self).__init__(backend.atan(inplace))
		self.inplace = inplace

class atanh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(atanh, self).__init__(backend.atanh(inplace))
		self.inplace = inplace

class cos(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(cos, self).__init__(backend.cos(inplace))
		self.inplace = inplace

class cosh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(cosh, self).__init__(backend.cosh(inplace))
		self.inplace = inplace

class sin(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(sin, self).__init__(backend.sin(inplace))
		self.inplace = inplace

class sinh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(sinh, self).__init__(backend.sinh(inplace))
		self.inplace = inplace

class tan(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(tan, self).__init__(backend.tan(inplace))
		self.inplace = inplace

class tanh(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(tanh, self).__init__(backend.tanh(inplace))
		self.inplace = inplace

class add(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(add, self).__init__(backend.add(inplace))
		self.inplace = inplace

class sub(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(sub, self).__init__(backend.sub(inplace))
		self.inplace = inplace

class mul(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(mul, self).__init__(backend.mul(inplace))
		self.inplace = inplace

class div(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(div, self).__init__(backend.div(inplace))
		self.inplace = inplace

class mod(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(mod, self).__init__(backend.mod(inplace))
		self.inplace = inplace

class pow(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(pow, self).__init__(backend.pow(inplace))
		self.inplace = inplace

class max(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(max, self).__init__(backend.max(inplace))
		self.inplace = inplace

class min(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(min, self).__init__(backend.min(inplace))
		self.inplace = inplace

class eq(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(eq, self).__init__(backend.eq(inplace))
		self.inplace = inplace

class ne(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(ne, self).__init__(backend.ne(inplace))
		self.inplace = inplace

class lt(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(lt, self).__init__(backend.lt(inplace))
		self.inplace = inplace

class le(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(le, self).__init__(backend.le(inplace))
		self.inplace = inplace

class gt(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(gt, self).__init__(backend.gt(inplace))
		self.inplace = inplace

class ge(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(ge, self).__init__(backend.ge(inplace))
		self.inplace = inplace

class xr(Module):
	__constants__ = ['inplace']
	inplace : bool

	def __init__(self, inplace: bool=False) -> None:
		super(xr, self).__init__(backend.xr(inplace))
		self.inplace = inplace