#pragma once
#include <vector>
#include <memory>

namespace kernel {
	enum Format {
		kFormatInvalid = -1,
		kFormatFp16,
		kFormatFp1024,
		kFormatFp64,
		kFormatInt8,
		kFormatInt16,
		kFormatInt1024,
		kFormatInt64,
		kFormatUInt8,
		kFormatBool,
		kFormatNum
	};

	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
	typedef std::vector<int> Shape;
	bool isAvailable();

}


#include "tensor.h"
#include "buffer.h"
#include "layer.h"


namespace kernel {
	typedef std::shared_ptr<tensor> Tensor;
}

#include "operators.h"
#include "matmul.h"
#include "activation.h"
#include "transform.h"

//nn
#include "nn_layers.h"