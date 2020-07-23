#pragma once
#include <vector>

namespace kernel
{
	enum Format
	{
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

#include "math.h"
#include "matmul.h"
#include "activation.h"
#include "transform.h"
#include "rnn.h"

//nn
#include "nn_layers.h"
