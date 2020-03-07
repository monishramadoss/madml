#pragma once
#include <vector>

namespace kernel {
	enum Format {
		kFormatInvalid = -1,
		kFormatFp16,
		kFormatFp32,
		kFormatFp64,
		kFormatInt8,
		kFormatInt16,
		kFormatInt32,
		kFormatInt64,
		kFormatUInt8,
		kFormatBool,
		kFormatNum
	};

	enum PaddingMode { kPaddingModeSame, kPaddingModeValid, kPaddingModeNum };
	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
	typedef std::vector<int> Shape;

	bool isAvailable();
}

#include "tensor.h"
#include "buffer.h"
#include "layer.h"

//
//
//
//#include "layers/abs.h"
//#include "layers/acos.h"
//#include "layers/acosh.h"
//#include "layers/add.h"
//#include "layers/and.h"
//#include "layers/asin.h"
//#include "layers/asinh.h"
//#include "layers/atan.h"
//#include "layers/atanh.h"
//
//#include "layers/ceil.h"
//#include "layers/clip.h"
//#include "layers/cos.h"
//#include "layers/cosh.h"
//
//
//#include "layers/elu.h"
//#include "layers/equal.h"
//#include "layers/exp.h"
//
//#include "layers/floor.h"
//
//#include "layers/greater.h"
//
//#include "layers/hardsigmoid.h"
//
//#include "layers/leakyrelu.h"
//#include "layers/less.h"
//#include "layers/log.h"
//
//#include "layers/max.h"
//#include "layers/min.h"
//#include "layers/mod.h"
//#include "layers/mul.h"
//
//#include "layers/neg.h"
//#include "layers/not.h"
//
//#include "layers/or.h"
//
//#include "layers/pow.h"
//#include "layers/prelu.h"
//
//#include "layers/reciprocal.h"
//#include "layers/relu.h"
//#include "layers/round.h"
//
//#include "layers/selu.h"
//#include "layers/sigmoid.h"
//#include "layers/sin.h"
//#include "layers/sinh.h"
//#include "layers/softplus.h"
//#include "layers/softsign.h"
//#include "layers/sqrt.h"
//#include "layers/sub.h"
//
//#include "layers/tan.h"
//#include "layers/tanh.h"
//
//#include "layers/xor.h"
//
//
