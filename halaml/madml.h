#pragma once
#include <vector>

namespace kernel
{
	enum class Format
	{
		kFormatInvalid = -1,
		kFormatFp16 = 2,
		kFormatFp32 = 4,
		kFormatFp64 = 8,
		kFormatInt8 = 1,
		kFormatInt16 = 2,
		kFormatInt32 = 4,
		kFormatInt64 = 8,
		kFormatUInt8 = 1,
		kFormatBool = 4,
		kFormatNum = -1
	};

	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };

	typedef std::vector<int> Shape;
	bool isAvailable();
}

//TODO need work on deallocations
//TODO work on static members
//TODO worker threads for first hand latency

#include "tensor.h"
#include "buffer.h"
#include "layer.h"

#include "./math.h"
#include "matmul.h"
#include "activation.h"
#include "transform.h"

//nn
#include "nn_layers.h"
