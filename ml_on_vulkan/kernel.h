#pragma once
#include <vector>
#include <memory>

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

	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
	typedef std::vector<int> Shape;
	bool isAvailable();

}


#include "tensor.h"
#include "buffer.h"
#include "layer.h"


namespace kernel {
	static std::vector<layer*> cmd_layer;
	static void run() {
		for (int i = 0; i < cmd_layer.size(); ++i) {
			cmd_layer[i]->run();
		}
	}
	typedef std::shared_ptr<tensor> Tensor;
}
#include "operators.h"
//nn
#include "convolution.h"